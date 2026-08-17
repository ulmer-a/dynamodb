use std::collections::HashMap;
use std::sync::Mutex;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{
    AnyIndex, AwsDynamoDbService, Error, INITIAL_DATA_VERSION, PrimaryIndex, SortKeyCondition,
};

/// In-memory [`AwsDynamoDbService`] implementation for local running and unit testing.
///
/// Payloads are stored as `serde_json::Value` keyed by `(table, pk, sk)` alongside their
/// `data_version`, mirroring the [`crate::Container`] layout without requiring a real
/// DynamoDB backend.
///
/// All tables in one store share the [`PrimaryIndex`] given to [`MockDynamoDb::new`], which is
/// what every operation validates its `sk` argument against — the same check the AWS backend
/// performs, so a key that is rejected by one implementation is rejected by the other.
#[derive(Debug)]
pub struct MockDynamoDb {
    primary_index: PrimaryIndex,
    items: Mutex<HashMap<MockDbKey, (u64, serde_json::Value)>>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MockDbKey {
    table_name: String,
    pk: String,
    sk: Option<String>,
}

impl MockDynamoDb {
    /// Creates an empty mock store whose tables use `primary_index` as their primary key
    /// schema.
    pub fn new(primary_index: PrimaryIndex) -> Self {
        Self {
            primary_index,
            items: Mutex::new(HashMap::new()),
        }
    }

    /// Inserts (or replaces) an item under `(pk, sk)` in `table` with the given `data_version`
    /// and payload.
    ///
    /// Intended for seeding test fixtures, including states the trait's own operations cannot
    /// produce — an arbitrary `data_version`, or `0` to stand in for a legacy record written
    /// before versioning existed.
    pub fn insert<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
        data_version: u64,
        payload: &T,
    ) -> Result<(), Error>
    where
        T: Serialize,
    {
        self.primary_index.resolve_sk(&sk)?;

        let value = serde_json::to_value(payload)
            .map_err(|e| Error::Serialization(format!("Failed to serialize mock payload: {e}")))?;
        self.items
            .lock()
            .map_err(|e| Error::Service(format!("mock store lock poisoned: {e}")))?
            .insert(
                MockDbKey {
                    table_name: table.to_string(),
                    pk,
                    sk,
                },
                (data_version, value),
            );
        Ok(())
    }
}

#[async_trait::async_trait]
impl AwsDynamoDbService for MockDynamoDb {
    async fn get_item<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
    ) -> Result<Option<(u64, T)>, Error>
    where
        T: DeserializeOwned + Send,
    {
        self.primary_index.resolve_sk(&sk)?;

        let guard = self
            .items
            .lock()
            .map_err(|e| Error::Service(format!("mock store lock poisoned: {e}")))?;

        let mock_key = MockDbKey {
            table_name: table.to_string(),
            pk: pk.clone(),
            sk: sk.clone(),
        };

        match guard.get(&mock_key) {
            Some((data_version, value)) => {
                let payload: T = serde_json::from_value(value.clone()).map_err(|e| {
                    Error::Serialization(format!(
                        "Failed to deserialize mock item for pk={pk}, sk={sk:?}: {e}"
                    ))
                })?;
                Ok(Some((*data_version, payload)))
            }
            None => Ok(None),
        }
    }

    async fn create_item<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
        payload: &T,
    ) -> Result<u64, Error>
    where
        T: Serialize + Send + Sync,
    {
        self.primary_index.resolve_sk(&sk)?;

        let value = serde_json::to_value(payload)
            .map_err(|e| Error::Serialization(format!("Failed to serialize mock payload: {e}")))?;

        let mut guard = self
            .items
            .lock()
            .map_err(|e| Error::Service(format!("mock store lock poisoned: {e}")))?;

        let map_key = MockDbKey {
            table_name: table.to_string(),
            pk: pk.clone(),
            sk: sk.clone(),
        };

        // Mirrors `attribute_not_exists(<pk>)`: the presence of the item is what's tested, not
        // its version, so a legacy record stored at version 0 also blocks the create.
        if guard.contains_key(&map_key) {
            return Err(Error::AlreadyExists { pk, sk });
        }

        guard.insert(map_key, (INITIAL_DATA_VERSION, value));
        Ok(INITIAL_DATA_VERSION)
    }

    async fn put_item_unconditional<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
        payload: &T,
    ) -> Result<u64, Error>
    where
        T: Serialize + Send + Sync,
    {
        self.primary_index.resolve_sk(&sk)?;

        let value = serde_json::to_value(payload)
            .map_err(|e| Error::Serialization(format!("Failed to serialize mock payload: {e}")))?;

        let mut guard = self
            .items
            .lock()
            .map_err(|e| Error::Service(format!("mock store lock poisoned: {e}")))?;

        let map_key = MockDbKey {
            table_name: table.to_string(),
            pk,
            sk,
        };

        // Last-writer-wins on the payload, but the version still advances: rewinding it would
        // make a spent version token valid again. The whole read-modify-write happens under the
        // store's lock, so unlike the AWS backend there's no window to retry around.
        let new_version = guard
            .get(&map_key)
            .map_or(INITIAL_DATA_VERSION, |(version, _)| version + 1);
        guard.insert(map_key, (new_version, value));
        Ok(new_version)
    }

    async fn put_item<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
        expected_version: u64,
        payload: &T,
    ) -> Result<u64, Error>
    where
        T: Serialize + Send + Sync,
    {
        self.primary_index.resolve_sk(&sk)?;

        let value = serde_json::to_value(payload)
            .map_err(|e| Error::Serialization(format!("Failed to serialize mock payload: {e}")))?;

        let mut guard = self
            .items
            .lock()
            .map_err(|e| Error::Service(format!("mock store lock poisoned: {e}")))?;

        let map_key = MockDbKey {
            table_name: table.to_string(),
            pk: pk.clone(),
            sk: sk.clone(),
        };
        let current_version = guard.get(&map_key).map(|(version, _)| *version);

        // Optimistic concurrency: the stored version must match what the caller expects,
        // otherwise a concurrent writer has changed the item. An absent item never matches, so
        // this can't create one — including for expected_version 0, which means "exists but
        // unversioned" rather than "does not exist".
        if current_version != Some(expected_version) {
            return Err(Error::Conflict {
                pk,
                sk,
                expected_version,
            });
        }

        // An unversioned record (0) lands at INITIAL_DATA_VERSION, upgrading it onto the scheme.
        let new_version = expected_version + 1;
        guard.insert(map_key, (new_version, value));
        Ok(new_version)
    }

    async fn add_to_counter(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
        counter_attribute: &str,
        delta: i64,
    ) -> Result<i64, Error> {
        self.primary_index
            .validate_counter_attribute(counter_attribute)?;
        self.primary_index.resolve_sk(&sk)?;

        let mut guard = self
            .items
            .lock()
            .map_err(|e| Error::Service(format!("mock store lock poisoned: {e}")))?;

        // DynamoDB's ADD creates the item when the key is absent, treating every attribute it
        // touches as having been 0. The whole read-modify-write happens under the store's lock,
        // which is how the mock gets the atomicity a single UpdateItem gives the AWS backend.
        let (version, value) = guard
            .entry(MockDbKey {
                table_name: table.to_string(),
                pk: pk.clone(),
                sk,
            })
            .or_insert_with(|| (0, serde_json::Value::Object(serde_json::Map::new())));

        let Some(object) = value.as_object_mut() else {
            return Err(Error::Service(format!(
                "Stored mock item for pk={pk} is not an object, so {counter_attribute:?} cannot \
                 be added to"
            )));
        };

        let current = match object.get(counter_attribute) {
            None => 0,
            Some(existing) => existing.as_i64().ok_or_else(|| {
                // DynamoDB rejects an ADD against a non-numeric attribute as a malformed
                // update, which surfaces as Error::Service — matched here so both backends
                // report it the same way.
                Error::Service(format!(
                    "Counter {counter_attribute:?} for pk={pk} is not a number: {existing}"
                ))
            })?,
        };

        let new_value = current + delta;
        object.insert(counter_attribute.to_string(), new_value.into());
        *version += 1;
        Ok(new_value)
    }

    async fn query_items<T>(
        &self,
        index: impl Into<AnyIndex> + Send,
        pk: String,
        sk_condition: Option<SortKeyCondition>,
        table: &str,
    ) -> Result<Vec<(u64, T)>, Error>
    where
        T: DeserializeOwned + Send,
    {
        let index = index.into();

        if sk_condition.is_some() && index.sk_identifier().is_none() {
            return Err(Error::sk_condition_without_sort_key());
        }
        if let Some(condition) = &sk_condition
            && !matches!(
                condition,
                SortKeyCondition::Prefix(_) | SortKeyCondition::Between(_, _)
            )
        {
            return Err(Error::Unsupported(format!(
                "{condition:?} is not yet implemented by MockDynamoDb"
            )));
        }

        let guard = self
            .items
            .lock()
            .map_err(|e| Error::Service(format!("mock store lock poisoned: {e}")))?;

        let mut matches: Vec<(String, u64, T)> = Vec::new();

        for (key, (data_version, value)) in guard.iter() {
            if key.table_name != table {
                continue;
            }

            // The primary index's key values live in the mock's own key. GSI key values are
            // regular attributes on the stored payload, as they would be on a real DynamoDB
            // item. LSIs share the base table's partition key (also the mock's own key) but
            // have their own sort key attribute, just like a GSI's.
            let (item_pk, item_sk) = match &index {
                AnyIndex::Primary(_) => (key.pk.clone(), key.sk.clone()),
                AnyIndex::Lsi(lsi) => {
                    let item_sk = value
                        .get(&lsi.sk_identifier)
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    (key.pk.clone(), item_sk)
                }
                AnyIndex::Gsi(gsi) => {
                    let Some(item_pk) = value.get(&gsi.keys.pk_identifier).and_then(|v| v.as_str())
                    else {
                        continue;
                    };
                    let item_sk = gsi
                        .keys
                        .sk_identifier
                        .as_deref()
                        .and_then(|id| value.get(id))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    (item_pk.to_string(), item_sk)
                }
            };

            if item_pk != pk {
                continue;
            }
            if let Some(condition) = &sk_condition {
                let sk_matches = match condition {
                    SortKeyCondition::Prefix(prefix) => item_sk
                        .as_deref()
                        .is_some_and(|sk| sk.starts_with(prefix.as_str())),
                    SortKeyCondition::Between(low, high) => item_sk
                        .as_deref()
                        .is_some_and(|sk| sk >= low.as_str() && sk <= high.as_str()),
                    _ => unreachable!("unimplemented variants are rejected before this loop"),
                };
                if !sk_matches {
                    continue;
                }
            }

            let payload: T = serde_json::from_value(value.clone()).map_err(|e| {
                Error::Serialization(format!("Failed to deserialize mock item for pk={pk}: {e}"))
            })?;
            matches.push((item_sk.unwrap_or_default(), *data_version, payload));
        }

        // DynamoDB returns query results ordered by sort key; sort so the mock behaves the same.
        matches.sort_by(|(a, _, _), (b, _, _)| a.cmp(b));

        Ok(matches
            .into_iter()
            .map(|(_, data_version, payload)| (data_version, payload))
            .collect())
    }

    async fn delete_item(&self, pk: String, sk: Option<String>, table: &str) -> Result<(), Error> {
        self.primary_index.resolve_sk(&sk)?;

        // DynamoDB DeleteItem is idempotent, so removing a missing key is a no-op success.
        self.items
            .lock()
            .map_err(|e| Error::Service(format!("mock store lock poisoned: {e}")))?
            .remove(&MockDbKey {
                table_name: table.to_string(),
                pk,
                sk,
            });
        Ok(())
    }
}

/// Behaviour that only the mock has, and so has no place in the shared conformance suite: the
/// [`MockDynamoDb::insert`] fixture helper. Everything the mock shares with the AWS backend is
/// tested in [`crate::conformance`], which runs the same assertions against both.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::KeySchema;
    use serde::Deserialize;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Payload {
        name: String,
        count: u32,
    }

    fn db() -> MockDynamoDb {
        MockDynamoDb::new(PrimaryIndex {
            keys: KeySchema {
                pk_identifier: "pk".to_string(),
                sk_identifier: Some("sk".to_string()),
            },
        })
    }

    fn pk_sk(pk: &str, sk: &str) -> (String, Option<String>) {
        (pk.to_string(), Some(sk.to_string()))
    }

    #[tokio::test]
    async fn insert_seeds_an_arbitrary_data_version() {
        let db = db();
        let payload = Payload {
            name: "balu".to_string(),
            count: 3,
        };
        let (pk, sk) = pk_sk("device#1", "METADATA");
        db.insert(pk.clone(), sk.clone(), "BaluCoreTable", 7, &payload)
            .unwrap();

        let result: Option<(u64, Payload)> = db.get_item(pk, sk, "BaluCoreTable").await.unwrap();

        assert_eq!(result, Some((7, payload)));
    }

    #[tokio::test]
    async fn insert_replaces_existing_item() {
        let db = db();
        let (pk, sk) = pk_sk("device#1", "METADATA");
        db.insert(
            pk.clone(),
            sk.clone(),
            "BaluCoreTable",
            1,
            &Payload {
                name: "old".to_string(),
                count: 1,
            },
        )
        .unwrap();
        let updated = Payload {
            name: "new".to_string(),
            count: 2,
        };
        db.insert(pk.clone(), sk.clone(), "BaluCoreTable", 2, &updated)
            .unwrap();

        let result: Option<(u64, Payload)> = db.get_item(pk, sk, "BaluCoreTable").await.unwrap();

        assert_eq!(result, Some((2, updated)));
    }

    /// Version `0` is what a record written before versioning existed reads back as (see
    /// [`crate::Container::data_version`]). The trait's own operations can't produce that state,
    /// so seeding it is the only way to exercise legacy records.
    #[tokio::test]
    async fn insert_can_seed_a_legacy_record_at_version_zero() {
        let db = db();
        let payload = Payload {
            name: "legacy".to_string(),
            count: 1,
        };
        let (pk, sk) = pk_sk("device#1", "METADATA");
        db.insert(pk.clone(), sk.clone(), "BaluCoreTable", 0, &payload)
            .unwrap();

        let result: Option<(u64, Payload)> = db
            .get_item(pk.clone(), sk.clone(), "BaluCoreTable")
            .await
            .unwrap();
        assert_eq!(result, Some((0, payload.clone())));

        // create_item won't clobber it: it tests whether the item exists, not what version it
        // holds.
        let err = db
            .create_item(pk.clone(), sk.clone(), "BaluCoreTable", &payload)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::AlreadyExists { .. }), "{err:?}");

        let result: Option<(u64, Payload)> = db
            .get_item(pk.clone(), sk.clone(), "BaluCoreTable")
            .await
            .unwrap();
        assert_eq!(result, Some((0, payload.clone())));

        // Feeding the version 0 that get_item reported straight back into put_item upgrades the
        // record onto the versioning scheme — no special case in the caller's loop.
        let upgraded = Payload {
            name: "upgraded".to_string(),
            count: 2,
        };
        let version = db
            .put_item(pk.clone(), sk.clone(), "BaluCoreTable", 0, &upgraded)
            .await
            .unwrap();
        assert_eq!(version, INITIAL_DATA_VERSION);

        let result: Option<(u64, Payload)> = db
            .get_item(pk.clone(), sk.clone(), "BaluCoreTable")
            .await
            .unwrap();
        assert_eq!(result, Some((1, upgraded.clone())));

        // Once upgraded, version 0 no longer matches: the record is versioned like any other.
        let err = db
            .put_item(pk, sk, "BaluCoreTable", 0, &payload)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Conflict { .. }), "{err:?}");
    }

    #[tokio::test]
    async fn insert_rejects_sort_key_not_matching_schema() {
        let db = db();
        let payload = Payload {
            name: "balu".to_string(),
            count: 1,
        };

        let err = db
            .insert("device#1".to_string(), None, "BaluCoreTable", 1, &payload)
            .unwrap_err();

        assert!(
            matches!(&err, Error::InvalidRequest(message)
                if message.contains("PrimaryIndex has a sort key")),
            "{err:?}"
        );
    }
}
