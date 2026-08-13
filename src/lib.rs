use serde::{Deserialize, Serialize};

#[cfg(feature = "aws")]
pub mod aws;

#[cfg(feature = "mock")]
pub mod mock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Keys {
    /// DynamoDB Partition Key. Must not be renamed.
    pub pk: String,

    /// DynamoDB Sort Key. Must not be renamed.
    pub sk: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Container<T> {
    // Partition and Sort Key
    #[serde(flatten)]
    keys: Keys,

    /// Data version used for optimistic concurrency control. Must not be renamed.
    /// This will default to zero if the version number is not present on the record
    /// (indicating a legacy record).
    #[serde(default)]
    data_version: u64,

    /// The actual payload data
    #[serde(flatten)]
    payload: T,
}

#[async_trait::async_trait]
pub trait AwsDynamoDbService: Send + Sync {
    /// DynamoDB get_item() operation.
    ///
    /// Reads the [`Container`] stored under `keys` and, if present, returns the stored
    /// `data_version` (for optimistic concurrency control) together with the deserialized
    /// payload.
    async fn get_item<T>(&self, keys: Keys, table: &str) -> Result<Option<(u64, T)>, String>
    where
        T: serde::de::DeserializeOwned + Send;

    /// DynamoDB put_item() operation without optimistic concurrency control.
    ///
    /// Unconditionally writes `payload` under `keys` in `table`, overwriting whatever item is
    /// currently stored (if any) and ignoring any existing `data_version`. The stored
    /// `data_version` is reset to `1`.
    ///
    /// Use this only for writes where last-writer-wins is acceptable; prefer
    /// [`AwsDynamoDbService::put_item`] when concurrent writers must not clobber each other.
    async fn put_item_unconditional<T>(
        &self,
        keys: Keys,
        table: &str,
        payload: &T,
    ) -> Result<(), String>
    where
        T: Serialize + Send + Sync;

    /// DynamoDB put_item() operation with optimistic concurrency control.
    ///
    /// Writes `payload` under `keys` in `table`, but only if the currently stored
    /// `data_version` matches `expected_version` (which must be > 0). Does not accept `0`
    /// (returns an error if passed).
    ///
    /// On success the version is incremented and the new `data_version` is returned. If the
    /// version check fails (a concurrent writer modified or created the item), an `Err` is
    /// returned and the table is left untouched.
    async fn put_item<T>(
        &self,
        keys: Keys,
        table: &str,
        expected_version: u64,
        payload: &T,
    ) -> Result<u64, String>
    where
        T: Serialize + Send + Sync;

    /// DynamoDB Query operation matching a sort-key prefix.
    ///
    /// Returns every [`Container`] stored under partition key `pk` whose sort key starts with
    /// `sk_prefix` (a DynamoDB `begins_with` query). For each matching item the stored
    /// `data_version` (for optimistic concurrency control) is returned together with the
    /// deserialized payload. An empty `Vec` is returned when nothing matches.
    async fn query_items_by_prefix<T>(
        &self,
        pk: String,
        sk_prefix: String,
        table: &str,
    ) -> Result<Vec<(u64, T)>, String>
    where
        T: serde::de::DeserializeOwned + Send;

    /// DynamoDB delete_item() operation.
    ///
    /// Removes the [`Container`] stored under `keys` in `table`. The operation is idempotent:
    /// deleting a key that does not exist succeeds and is treated as a no-op (mirroring
    /// DynamoDB's DeleteItem semantics). On success `Ok(())` is returned; transport or
    /// service errors are surfaced as an `Err` describing the failure.
    async fn delete_item(&self, keys: Keys, table: &str) -> Result<(), String>;
}

#[cfg(test)]
mod tests {
    use serde::Serialize;
    use std::collections::HashMap;

    use crate::{Container, Keys};

    #[test]
    fn test_container_fields_not_renamed() {
        #[derive(Serialize)]
        struct TestPayload {
            magic: u32,
        }

        let container = Container {
            keys: Keys {
                pk: "foo".into(),
                sk: "bar".into(),
            },
            data_version: 5,
            payload: TestPayload { magic: 42 },
        };

        let map: HashMap<String, serde_json::Value> =
            serde_json::from_value(serde_json::to_value(&container).unwrap()).unwrap();

        assert!(map.contains_key("pk"));
        assert!(map.contains_key("sk"));
        assert!(map.contains_key("data_version"));
    }
}
