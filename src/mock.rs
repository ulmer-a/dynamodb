use std::collections::HashMap;
use std::sync::Mutex;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{AwsDynamoDbService, Keys};

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
    sk: String,
}

impl MockDynamoDb {
    /// Creates an empty mock store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts (or replaces) an item under `keys` in `table` with the given `data_version`
    /// and payload.
    ///
    /// Intended for seeding test fixtures.
    pub fn insert<T>(
        &self,
        keys: Keys,
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
                    pk: keys.pk,
                    sk: keys.sk,
                },
                (data_version, value),
            );
        Ok(())
    }
}

#[async_trait::async_trait]
impl AwsDynamoDbService for MockDynamoDb {
    async fn get_item<T>(&self, keys: Keys, table: &str) -> Result<Option<(u64, T)>, String>
    where
        T: DeserializeOwned + Send,
    {
        let guard = self
            .items
            .lock()
            .map_err(|e| format!("mock store lock poisoned: {e}"))?;

        let mock_key = MockDbKey {
            table_name: table.to_string(),
            pk: keys.pk.clone(),
            sk: keys.sk.clone(),
        };

        match guard.get(&mock_key) {
            Some((data_version, value)) => {
                let payload: T = serde_json::from_value(value.clone()).map_err(|e| {
                    format!(
                        "Failed to deserialize mock item for pk={}, sk={}: {e}",
                        keys.pk, keys.sk
                    )
                })?;
                Ok(Some((*data_version, payload)))
            }
            None => Ok(None),
        }
    }

    async fn put_item_unconditional<T>(
        &self,
        keys: Keys,
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
                    pk: keys.pk,
                    sk: keys.sk,
                },
                (1, value),
            );
        Ok(())
    }

    async fn put_item<T>(
        &self,
        keys: Keys,
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
            pk: keys.pk.clone(),
            sk: keys.sk.clone(),
        };
        let current_version = guard.get(&map_key).map(|(version, _)| *version);

        // Optimistic concurrency: the stored version (or its absence) must match what the
        // caller expects, otherwise a concurrent writer has changed the item.
        if current_version != Some(expected_version) {
            return Err(format!(
                "Optimistic concurrency conflict for pk={}, sk={}: expected version {expected_version}, found {current_version:?}",
                keys.pk, keys.sk
            ));
        }

        let new_version = expected_version + 1;
        guard.insert(map_key, (new_version, value));
        Ok(new_version)
    }

    async fn query_items_by_prefix<T>(
        &self,
        pk: String,
        sk_prefix: String,
        table: &str,
    ) -> Result<Vec<(u64, T)>, String>
    where
        T: DeserializeOwned + Send,
    {
        let guard = self
            .items
            .lock()
            .map_err(|e| format!("mock store lock poisoned: {e}"))?;

        let mut matches: Vec<(String, u64, T)> = guard
            .iter()
            .filter(|(key, _)| {
                key.table_name == table && key.pk == pk && key.sk.starts_with(&sk_prefix)
            })
            .map(|(key, (data_version, value))| {
                let payload: T = serde_json::from_value(value.clone()).map_err(|e| {
                    format!(
                        "Failed to deserialize mock item for pk={pk}, sk={}: {e}",
                        key.sk
                    )
                })?;
                Ok((key.sk.clone(), *data_version, payload))
            })
            .collect::<Result<_, String>>()?;

        // DynamoDB returns query results ordered by sort key; sort so the mock behaves the same.
        matches.sort_by(|(a, _, _), (b, _, _)| a.cmp(b));

        Ok(matches
            .into_iter()
            .map(|(_, data_version, payload)| (data_version, payload))
            .collect())
    }

    async fn delete_item(&self, keys: Keys, table: &str) -> Result<(), String> {
        // DynamoDB DeleteItem is idempotent, so removing a missing key is a no-op success.
        self.items
            .lock()
            .map_err(|e| format!("mock store lock poisoned: {e}"))?
            .remove(&MockDbKey {
                table_name: table.to_string(),
                pk: keys.pk,
                sk: keys.sk,
            });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Payload {
        name: String,
        count: u32,
    }

    fn keys(pk: &str, sk: &str) -> Keys {
        Keys {
            pk: pk.to_string(),
            sk: sk.to_string(),
        }
    }

    #[tokio::test]
    async fn get_item_returns_inserted_payload_with_version() {
        let db = MockDynamoDb::new();
        let payload = Payload {
            name: "balu".to_string(),
            count: 3,
        };
        db.insert(keys("device#1", "METADATA"), "BaluCoreTable", 7, &payload)
            .unwrap();

        let result: Option<(u64, Payload)> = db
            .get_item(keys("device#1", "METADATA"), "BaluCoreTable")
            .await
            .unwrap();

        assert_eq!(result, Some((7, payload)));
    }

    #[tokio::test]
    async fn get_item_returns_none_for_missing_key() {
        let db = MockDynamoDb::new();

        let result: Option<(u64, Payload)> = db
            .get_item(keys("device#missing", "METADATA"), "BaluCoreTable")
            .await
            .unwrap();

        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn insert_replaces_existing_item() {
        let db = MockDynamoDb::new();
        db.insert(
            keys("device#1", "METADATA"),
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
        db.insert(keys("device#1", "METADATA"), "BaluCoreTable", 2, &updated)
            .unwrap();

        let result: Option<(u64, Payload)> = db
            .get_item(keys("device#1", "METADATA"), "BaluCoreTable")
            .await
            .unwrap();

        assert_eq!(result, Some((2, updated)));
    }

    #[tokio::test]
    async fn put_item_updates_when_version_matches() {
        let db = MockDynamoDb::new();
        db.insert(
            keys("device#1", "METADATA"),
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
            .put_item(keys("device#1", "METADATA"), "BaluCoreTable", 5, &updated)
            .await
            .unwrap();

        assert_eq!(new_version, 6);
        let result: Option<(u64, Payload)> = db
            .get_item(keys("device#1", "METADATA"), "BaluCoreTable")
            .await
            .unwrap();
        assert_eq!(result, Some((6, updated)));
    }

    #[tokio::test]
    async fn put_item_rejects_stale_version() {
        let db = MockDynamoDb::new();
        let original = Payload {
            name: "old".to_string(),
            count: 1,
        };
        db.insert(keys("device#1", "METADATA"), "BaluCoreTable", 5, &original)
            .unwrap();

        let err = db
            .put_item(
                keys("device#1", "METADATA"),
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
        let result: Option<(u64, Payload)> = db
            .get_item(keys("device#1", "METADATA"), "BaluCoreTable")
            .await
            .unwrap();
        assert_eq!(result, Some((5, original)));
    }

    #[tokio::test]
    async fn query_items_by_prefix_returns_matching_items_sorted_by_sort_key() {
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
        db.insert(keys("device#1", "EVENT#2"), "BaluCoreTable", 4, &second)
            .unwrap();
        db.insert(keys("device#1", "EVENT#1"), "BaluCoreTable", 3, &first)
            .unwrap();

        let result: Vec<(u64, Payload)> = db
            .query_items_by_prefix(
                "device#1".to_string(),
                "EVENT#".to_string(),
                "BaluCoreTable",
            )
            .await
            .unwrap();

        assert_eq!(result, vec![(3, first), (4, second)]);
    }

    #[tokio::test]
    async fn query_items_by_prefix_excludes_non_matching_keys() {
        let db = MockDynamoDb::new();
        let event = Payload {
            name: "event".to_string(),
            count: 1,
        };
        let metadata = Payload {
            name: "metadata".to_string(),
            count: 2,
        };
        db.insert(keys("device#1", "EVENT#1"), "BaluCoreTable", 1, &event)
            .unwrap();
        // Different sort-key prefix, same partition.
        db.insert(keys("device#1", "METADATA"), "BaluCoreTable", 1, &metadata)
            .unwrap();
        // Matching sort-key prefix but a different partition.
        db.insert(keys("device#2", "EVENT#1"), "BaluCoreTable", 1, &event)
            .unwrap();
        // Matching keys but a different table.
        db.insert(keys("device#1", "EVENT#1"), "OtherTable", 1, &event)
            .unwrap();

        let result: Vec<(u64, Payload)> = db
            .query_items_by_prefix(
                "device#1".to_string(),
                "EVENT#".to_string(),
                "BaluCoreTable",
            )
            .await
            .unwrap();

        assert_eq!(result, vec![(1, event)]);
    }

    #[tokio::test]
    async fn put_item_unconditional_creates_item_with_version_one() {
        let db = MockDynamoDb::new();
        let payload = Payload {
            name: "balu".to_string(),
            count: 1,
        };

        db.put_item_unconditional(keys("device#1", "METADATA"), "BaluCoreTable", &payload)
            .await
            .unwrap();

        let result: Option<(u64, Payload)> = db
            .get_item(keys("device#1", "METADATA"), "BaluCoreTable")
            .await
            .unwrap();
        assert_eq!(result, Some((1, payload)));
    }

    #[tokio::test]
    async fn put_item_unconditional_overwrites_existing_item_regardless_of_version() {
        let db = MockDynamoDb::new();
        db.insert(
            keys("device#1", "METADATA"),
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

        db.put_item_unconditional(keys("device#1", "METADATA"), "BaluCoreTable", &updated)
            .await
            .unwrap();

        // The pre-existing version is ignored and reset to 1 on an unconditional write.
        let result: Option<(u64, Payload)> = db
            .get_item(keys("device#1", "METADATA"), "BaluCoreTable")
            .await
            .unwrap();
        assert_eq!(result, Some((1, updated)));
    }

    #[tokio::test]
    async fn delete_item_removes_existing_item() {
        let db = MockDynamoDb::new();
        db.insert(
            keys("device#1", "METADATA"),
            "BaluCoreTable",
            1,
            &Payload {
                name: "balu".to_string(),
                count: 1,
            },
        )
        .unwrap();

        db.delete_item(keys("device#1", "METADATA"), "BaluCoreTable")
            .await
            .unwrap();

        let result: Option<(u64, Payload)> = db
            .get_item(keys("device#1", "METADATA"), "BaluCoreTable")
            .await
            .unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn delete_item_is_idempotent_for_missing_key() {
        let db = MockDynamoDb::new();

        // Deleting a key that was never inserted must succeed as a no-op.
        db.delete_item(keys("device#missing", "METADATA"), "BaluCoreTable")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn delete_item_only_removes_targeted_key() {
        let db = MockDynamoDb::new();
        let kept = Payload {
            name: "kept".to_string(),
            count: 1,
        };
        db.insert(keys("device#1", "EVENT#1"), "BaluCoreTable", 1, &kept)
            .unwrap();
        db.insert(
            keys("device#1", "EVENT#2"),
            "BaluCoreTable",
            1,
            &Payload {
                name: "gone".to_string(),
                count: 2,
            },
        )
        .unwrap();

        db.delete_item(keys("device#1", "EVENT#2"), "BaluCoreTable")
            .await
            .unwrap();

        let result: Option<(u64, Payload)> = db
            .get_item(keys("device#1", "EVENT#1"), "BaluCoreTable")
            .await
            .unwrap();
        assert_eq!(result, Some((1, kept)));
    }

    #[tokio::test]
    async fn query_items_by_prefix_returns_empty_when_no_match() {
        let db = MockDynamoDb::new();

        let result: Vec<(u64, Payload)> = db
            .query_items_by_prefix(
                "device#1".to_string(),
                "EVENT#".to_string(),
                "BaluCoreTable",
            )
            .await
            .unwrap();

        assert!(result.is_empty());
    }
}
