use std::collections::HashMap;
use std::sync::Mutex;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{AnyIndex, AwsDynamoDbService, SortKeyCondition};

/// In-memory [`AwsDynamoDbService`] implementation for local running and unit testing.
///
/// Payloads are stored as `serde_json::Value` keyed by `(table, pk, sk)` alongside their
/// `data_version`, mirroring the [`crate::Container`] layout without requiring a real
/// DynamoDB backend.
#[derive(Debug, Default)]
pub struct MockDynamoDb {
    items: Mutex<HashMap<MockDbKey, (u64, serde_json::Value)>>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MockDbKey {
    table_name: String,
    pk: String,
    sk: Option<String>,
}

impl MockDynamoDb {
    /// Creates an empty mock store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts (or replaces) an item under `(pk, sk)` in `table` with the given `data_version`
    /// and payload.
    ///
    /// Intended for seeding test fixtures.
    pub fn insert<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
        data_version: u64,
        payload: &T,
    ) -> Result<(), String>
    where
        T: Serialize,
    {
        let value = serde_json::to_value(payload)
            .map_err(|e| format!("Failed to serialize mock payload: {e}"))?;
        self.items
            .lock()
            .map_err(|e| format!("mock store lock poisoned: {e}"))?
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
    ) -> Result<Option<(u64, T)>, String>
    where
        T: DeserializeOwned + Send,
    {
        let guard = self
            .items
            .lock()
            .map_err(|e| format!("mock store lock poisoned: {e}"))?;

        let mock_key = MockDbKey {
            table_name: table.to_string(),
            pk: pk.clone(),
            sk: sk.clone(),
        };

        match guard.get(&mock_key) {
            Some((data_version, value)) => {
                let payload: T = serde_json::from_value(value.clone()).map_err(|e| {
                    format!("Failed to deserialize mock item for pk={pk}, sk={sk:?}: {e}")
                })?;
                Ok(Some((*data_version, payload)))
            }
            None => Ok(None),
        }
    }

    async fn put_item_unconditional<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
        payload: &T,
    ) -> Result<(), String>
    where
        T: Serialize + Send + Sync,
    {
        let value = serde_json::to_value(payload)
            .map_err(|e| format!("Failed to serialize mock payload: {e}"))?;

        // Last-writer-wins: overwrite any existing item and reset the version to 1.
        self.items
            .lock()
            .map_err(|e| format!("mock store lock poisoned: {e}"))?
            .insert(
                MockDbKey {
                    table_name: table.to_string(),
                    pk,
                    sk,
                },
                (1, value),
            );
        Ok(())
    }

    async fn put_item<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
        expected_version: u64,
        payload: &T,
    ) -> Result<u64, String>
    where
        T: Serialize + Send + Sync,
    {
        if expected_version == 0 {
            return Err("put_item does not accept expected version 0".to_string());
        }

        let value = serde_json::to_value(payload)
            .map_err(|e| format!("Failed to serialize mock payload: {e}"))?;

        let mut guard = self
            .items
            .lock()
            .map_err(|e| format!("mock store lock poisoned: {e}"))?;

        let map_key = MockDbKey {
            table_name: table.to_string(),
            pk: pk.clone(),
            sk: sk.clone(),
        };
        let current_version = guard.get(&map_key).map(|(version, _)| *version);

        // Optimistic concurrency: the stored version (or its absence) must match what the
        // caller expects, otherwise a concurrent writer has changed the item.
        if current_version != Some(expected_version) {
            return Err(format!(
                "Optimistic concurrency conflict for pk={pk}, sk={sk:?}: expected version {expected_version}, found {current_version:?}"
            ));
        }

        let new_version = expected_version + 1;
        guard.insert(map_key, (new_version, value));
        Ok(new_version)
    }

    async fn query_items<T>(
        &self,
        index: impl Into<AnyIndex> + Send,
        pk: String,
        sk_condition: Option<SortKeyCondition>,
        table: &str,
    ) -> Result<Vec<(u64, T)>, String>
    where
        T: DeserializeOwned + Send,
    {
        let index = index.into();

        if sk_condition.is_some() && index.sk_identifier().is_none() {
            return Err(
                "a sort key condition was provided but the index has no sort key".to_string(),
            );
        }
        if let Some(condition) = &sk_condition
            && !matches!(
                condition,
                SortKeyCondition::Prefix(_) | SortKeyCondition::Between(_, _)
            )
        {
            return Err(format!(
                "{condition:?} is not yet implemented by MockDynamoDb"
            ));
        }

        let guard = self
            .items
            .lock()
            .map_err(|e| format!("mock store lock poisoned: {e}"))?;

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

            let payload: T = serde_json::from_value(value.clone())
                .map_err(|e| format!("Failed to deserialize mock item for pk={pk}: {e}"))?;
            matches.push((item_sk.unwrap_or_default(), *data_version, payload));
        }

        // DynamoDB returns query results ordered by sort key; sort so the mock behaves the same.
        matches.sort_by(|(a, _, _), (b, _, _)| a.cmp(b));

        Ok(matches
            .into_iter()
            .map(|(_, data_version, payload)| (data_version, payload))
            .collect())
    }

    async fn delete_item(&self, pk: String, sk: Option<String>, table: &str) -> Result<(), String> {
        // DynamoDB DeleteItem is idempotent, so removing a missing key is a no-op success.
        self.items
            .lock()
            .map_err(|e| format!("mock store lock poisoned: {e}"))?
            .remove(&MockDbKey {
                table_name: table.to_string(),
                pk,
                sk,
            });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GlobalSecondaryIndex, KeySchema, LocalSecondaryIndex, PrimaryIndex};
    use serde::Deserialize;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Payload {
        name: String,
        count: u32,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct PayloadWithGsi {
        gsi1pk: String,
        gsi1sk: String,
        name: String,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct PayloadWithLsi {
        lsi1sk: String,
        name: String,
    }

    fn pk_sk(pk: &str, sk: &str) -> (String, Option<String>) {
        (pk.to_string(), Some(sk.to_string()))
    }

    // This project's tables happen to use "pk"/"sk", but nothing in the mock or the real AWS
    // implementation requires that convention — it's just what this test suite picked.
    fn primary_index() -> PrimaryIndex {
        PrimaryIndex {
            keys: KeySchema {
                pk_identifier: "pk".to_string(),
                sk_identifier: Some("sk".to_string()),
            },
        }
    }

    #[tokio::test]
    async fn get_item_returns_inserted_payload_with_version() {
        let db = MockDynamoDb::new();
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
    async fn get_item_returns_none_for_missing_key() {
        let db = MockDynamoDb::new();
        let (pk, sk) = pk_sk("device#missing", "METADATA");

        let result: Option<(u64, Payload)> = db.get_item(pk, sk, "BaluCoreTable").await.unwrap();

        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn insert_replaces_existing_item() {
        let db = MockDynamoDb::new();
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

    #[tokio::test]
    async fn put_item_updates_when_version_matches() {
        let db = MockDynamoDb::new();
        let (pk, sk) = pk_sk("device#1", "METADATA");
        db.insert(
            pk.clone(),
            sk.clone(),
            "BaluCoreTable",
            5,
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

        let new_version = db
            .put_item(pk.clone(), sk.clone(), "BaluCoreTable", 5, &updated)
            .await
            .unwrap();

        assert_eq!(new_version, 6);
        let result: Option<(u64, Payload)> = db.get_item(pk, sk, "BaluCoreTable").await.unwrap();
        assert_eq!(result, Some((6, updated)));
    }

    #[tokio::test]
    async fn put_item_rejects_stale_version() {
        let db = MockDynamoDb::new();
        let original = Payload {
            name: "old".to_string(),
            count: 1,
        };
        let (pk, sk) = pk_sk("device#1", "METADATA");
        db.insert(pk.clone(), sk.clone(), "BaluCoreTable", 5, &original)
            .unwrap();

        let err = db
            .put_item(
                pk.clone(),
                sk.clone(),
                "BaluCoreTable",
                4,
                &Payload {
                    name: "new".to_string(),
                    count: 2,
                },
            )
            .await
            .unwrap_err();

        assert!(err.contains("Optimistic concurrency conflict"), "{err}");
        // The stored item must be left untouched after a rejected write.
        let result: Option<(u64, Payload)> = db.get_item(pk, sk, "BaluCoreTable").await.unwrap();
        assert_eq!(result, Some((5, original)));
    }

    #[tokio::test]
    async fn query_items_returns_matching_items_sorted_by_sort_key() {
        let db = MockDynamoDb::new();
        let first = Payload {
            name: "first".to_string(),
            count: 1,
        };
        let second = Payload {
            name: "second".to_string(),
            count: 2,
        };
        // Inserted out of sort-key order to verify the result is sorted.
        db.insert(
            "device#1".to_string(),
            Some("EVENT#2".to_string()),
            "BaluCoreTable",
            4,
            &second,
        )
        .unwrap();
        db.insert(
            "device#1".to_string(),
            Some("EVENT#1".to_string()),
            "BaluCoreTable",
            3,
            &first,
        )
        .unwrap();

        let result: Vec<(u64, Payload)> = db
            .query_items(
                primary_index(),
                "device#1".to_string(),
                Some(SortKeyCondition::Prefix("EVENT#".to_string())),
                "BaluCoreTable",
            )
            .await
            .unwrap();

        assert_eq!(result, vec![(3, first), (4, second)]);
    }

    #[tokio::test]
    async fn query_items_excludes_non_matching_keys() {
        let db = MockDynamoDb::new();
        let event = Payload {
            name: "event".to_string(),
            count: 1,
        };
        let metadata = Payload {
            name: "metadata".to_string(),
            count: 2,
        };
        db.insert(
            "device#1".to_string(),
            Some("EVENT#1".to_string()),
            "BaluCoreTable",
            1,
            &event,
        )
        .unwrap();
        // Different sort-key prefix, same partition.
        db.insert(
            "device#1".to_string(),
            Some("METADATA".to_string()),
            "BaluCoreTable",
            1,
            &metadata,
        )
        .unwrap();
        // Matching sort-key prefix but a different partition.
        db.insert(
            "device#2".to_string(),
            Some("EVENT#1".to_string()),
            "BaluCoreTable",
            1,
            &event,
        )
        .unwrap();
        // Matching keys but a different table.
        db.insert(
            "device#1".to_string(),
            Some("EVENT#1".to_string()),
            "OtherTable",
            1,
            &event,
        )
        .unwrap();

        let result: Vec<(u64, Payload)> = db
            .query_items(
                primary_index(),
                "device#1".to_string(),
                Some(SortKeyCondition::Prefix("EVENT#".to_string())),
                "BaluCoreTable",
            )
            .await
            .unwrap();

        assert_eq!(result, vec![(1, event)]);
    }

    #[tokio::test]
    async fn query_items_via_gsi_uses_payload_attributes() {
        let db = MockDynamoDb::new();
        let gsi = GlobalSecondaryIndex {
            name: "GSI1".to_string(),
            keys: KeySchema {
                pk_identifier: "gsi1pk".to_string(),
                sk_identifier: Some("gsi1sk".to_string()),
            },
        };

        let matching = PayloadWithGsi {
            gsi1pk: "team#a".to_string(),
            gsi1sk: "user#1".to_string(),
            name: "matching".to_string(),
        };
        let other_team = PayloadWithGsi {
            gsi1pk: "team#b".to_string(),
            gsi1sk: "user#1".to_string(),
            name: "other team".to_string(),
        };
        let non_matching_sk = PayloadWithGsi {
            gsi1pk: "team#a".to_string(),
            gsi1sk: "device#1".to_string(),
            name: "non matching sk".to_string(),
        };

        db.insert(
            "device#1".to_string(),
            Some("METADATA".to_string()),
            "BaluCoreTable",
            1,
            &matching,
        )
        .unwrap();
        db.insert(
            "device#2".to_string(),
            Some("METADATA".to_string()),
            "BaluCoreTable",
            1,
            &other_team,
        )
        .unwrap();
        db.insert(
            "device#3".to_string(),
            Some("METADATA".to_string()),
            "BaluCoreTable",
            1,
            &non_matching_sk,
        )
        .unwrap();

        let result: Vec<(u64, PayloadWithGsi)> = db
            .query_items(
                gsi,
                "team#a".to_string(),
                Some(SortKeyCondition::Prefix("user#".to_string())),
                "BaluCoreTable",
            )
            .await
            .unwrap();

        assert_eq!(result, vec![(1, matching)]);
    }

    #[tokio::test]
    async fn query_items_via_lsi_shares_primary_partition_key() {
        let db = MockDynamoDb::new();
        let lsi = LocalSecondaryIndex {
            name: "LSI1".to_string(),
            sk_identifier: "lsi1sk".to_string(),
        };

        let matching = PayloadWithLsi {
            lsi1sk: "2026-01-01".to_string(),
            name: "matching".to_string(),
        };
        let other_partition = PayloadWithLsi {
            lsi1sk: "2026-01-01".to_string(),
            name: "other partition".to_string(),
        };

        // LSIs share the base table's partition key, so this item must be found by its
        // primary-key pk ("device#1"), not by any attribute inside the payload.
        db.insert(
            "device#1".to_string(),
            Some("METADATA".to_string()),
            "BaluCoreTable",
            1,
            &matching,
        )
        .unwrap();
        db.insert(
            "device#2".to_string(),
            Some("METADATA".to_string()),
            "BaluCoreTable",
            1,
            &other_partition,
        )
        .unwrap();

        let result: Vec<(u64, PayloadWithLsi)> = db
            .query_items(
                lsi,
                "device#1".to_string(),
                Some(SortKeyCondition::Prefix("2026".to_string())),
                "BaluCoreTable",
            )
            .await
            .unwrap();

        assert_eq!(result, vec![(1, matching)]);
    }

    #[tokio::test]
    async fn put_item_unconditional_creates_item_with_version_one() {
        let db = MockDynamoDb::new();
        let payload = Payload {
            name: "balu".to_string(),
            count: 1,
        };
        let (pk, sk) = pk_sk("device#1", "METADATA");

        db.put_item_unconditional(pk.clone(), sk.clone(), "BaluCoreTable", &payload)
            .await
            .unwrap();

        let result: Option<(u64, Payload)> = db.get_item(pk, sk, "BaluCoreTable").await.unwrap();
        assert_eq!(result, Some((1, payload)));
    }

    #[tokio::test]
    async fn put_item_unconditional_overwrites_existing_item_regardless_of_version() {
        let db = MockDynamoDb::new();
        let (pk, sk) = pk_sk("device#1", "METADATA");
        db.insert(
            pk.clone(),
            sk.clone(),
            "BaluCoreTable",
            42,
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

        db.put_item_unconditional(pk.clone(), sk.clone(), "BaluCoreTable", &updated)
            .await
            .unwrap();

        // The pre-existing version is ignored and reset to 1 on an unconditional write.
        let result: Option<(u64, Payload)> = db.get_item(pk, sk, "BaluCoreTable").await.unwrap();
        assert_eq!(result, Some((1, updated)));
    }

    #[tokio::test]
    async fn delete_item_removes_existing_item() {
        let db = MockDynamoDb::new();
        let (pk, sk) = pk_sk("device#1", "METADATA");
        db.insert(
            pk.clone(),
            sk.clone(),
            "BaluCoreTable",
            1,
            &Payload {
                name: "balu".to_string(),
                count: 1,
            },
        )
        .unwrap();

        db.delete_item(pk.clone(), sk.clone(), "BaluCoreTable")
            .await
            .unwrap();

        let result: Option<(u64, Payload)> = db.get_item(pk, sk, "BaluCoreTable").await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn delete_item_is_idempotent_for_missing_key() {
        let db = MockDynamoDb::new();
        let (pk, sk) = pk_sk("device#missing", "METADATA");

        // Deleting a key that was never inserted must succeed as a no-op.
        db.delete_item(pk, sk, "BaluCoreTable").await.unwrap();
    }

    #[tokio::test]
    async fn delete_item_only_removes_targeted_key() {
        let db = MockDynamoDb::new();
        let kept = Payload {
            name: "kept".to_string(),
            count: 1,
        };
        db.insert(
            "device#1".to_string(),
            Some("EVENT#1".to_string()),
            "BaluCoreTable",
            1,
            &kept,
        )
        .unwrap();
        db.insert(
            "device#1".to_string(),
            Some("EVENT#2".to_string()),
            "BaluCoreTable",
            1,
            &Payload {
                name: "gone".to_string(),
                count: 2,
            },
        )
        .unwrap();

        db.delete_item(
            "device#1".to_string(),
            Some("EVENT#2".to_string()),
            "BaluCoreTable",
        )
        .await
        .unwrap();

        let result: Option<(u64, Payload)> = db
            .get_item(
                "device#1".to_string(),
                Some("EVENT#1".to_string()),
                "BaluCoreTable",
            )
            .await
            .unwrap();
        assert_eq!(result, Some((1, kept)));
    }

    #[tokio::test]
    async fn query_items_returns_empty_when_no_match() {
        let db = MockDynamoDb::new();

        let result: Vec<(u64, Payload)> = db
            .query_items(
                primary_index(),
                "device#1".to_string(),
                Some(SortKeyCondition::Prefix("EVENT#".to_string())),
                "BaluCoreTable",
            )
            .await
            .unwrap();

        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn query_items_rejects_sk_condition_when_index_has_no_sort_key() {
        let db = MockDynamoDb::new();
        let gsi_without_sk = GlobalSecondaryIndex {
            name: "GSI2".to_string(),
            keys: KeySchema {
                pk_identifier: "gsi2pk".to_string(),
                sk_identifier: None,
            },
        };

        let err = db
            .query_items::<Payload>(
                gsi_without_sk,
                "device#1".to_string(),
                Some(SortKeyCondition::Prefix("EVENT#".to_string())),
                "BaluCoreTable",
            )
            .await
            .unwrap_err();

        assert!(err.contains("no sort key"), "{err}");
    }

    #[tokio::test]
    async fn query_items_via_between_condition() {
        let db = MockDynamoDb::new();
        let low = Payload {
            name: "low".to_string(),
            count: 1,
        };
        let mid = Payload {
            name: "mid".to_string(),
            count: 2,
        };
        let high = Payload {
            name: "high".to_string(),
            count: 3,
        };
        db.insert(
            "device#1".to_string(),
            Some("EVENT#1".to_string()),
            "BaluCoreTable",
            1,
            &low,
        )
        .unwrap();
        db.insert(
            "device#1".to_string(),
            Some("EVENT#2".to_string()),
            "BaluCoreTable",
            1,
            &mid,
        )
        .unwrap();
        db.insert(
            "device#1".to_string(),
            Some("EVENT#3".to_string()),
            "BaluCoreTable",
            1,
            &high,
        )
        .unwrap();

        let result: Vec<(u64, Payload)> = db
            .query_items(
                primary_index(),
                "device#1".to_string(),
                Some(SortKeyCondition::Between(
                    "EVENT#1".to_string(),
                    "EVENT#2".to_string(),
                )),
                "BaluCoreTable",
            )
            .await
            .unwrap();

        assert_eq!(result, vec![(1, low), (1, mid)]);
    }

    #[tokio::test]
    async fn query_items_rejects_unimplemented_sort_key_condition() {
        let db = MockDynamoDb::new();

        let err = db
            .query_items::<Payload>(
                primary_index(),
                "device#1".to_string(),
                Some(SortKeyCondition::Equals("METADATA".to_string())),
                "BaluCoreTable",
            )
            .await
            .unwrap_err();

        assert!(err.contains("not yet implemented"), "{err}");
    }

    #[tokio::test]
    async fn get_item_and_put_item_support_a_partition_only_primary_key() {
        let db = MockDynamoDb::new();
        let payload = Payload {
            name: "balu".to_string(),
            count: 1,
        };

        db.put_item_unconditional("device#1".to_string(), None, "BaluCoreTable", &payload)
            .await
            .unwrap();

        let result: Option<(u64, Payload)> = db
            .get_item("device#1".to_string(), None, "BaluCoreTable")
            .await
            .unwrap();
        assert_eq!(result, Some((1, payload.clone())));

        let updated = Payload {
            name: "updated".to_string(),
            count: 2,
        };
        let new_version = db
            .put_item("device#1".to_string(), None, "BaluCoreTable", 1, &updated)
            .await
            .unwrap();
        assert_eq!(new_version, 2);

        db.delete_item("device#1".to_string(), None, "BaluCoreTable")
            .await
            .unwrap();

        let result: Option<(u64, Payload)> = db
            .get_item("device#1".to_string(), None, "BaluCoreTable")
            .await
            .unwrap();
        assert_eq!(result, None);
    }
}
