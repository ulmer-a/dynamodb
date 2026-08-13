use serde::{Deserialize, Serialize};

#[cfg(feature = "aws")]
pub mod aws;

#[cfg(feature = "mock")]
pub mod mock;

/// The key attribute names of an index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeySchema {
    /// Attribute name of the partition key.
    pub pk_identifier: String,

    /// Attribute name of the sort key, if the index has one.
    pub sk_identifier: Option<String>,
}

/// The base table's primary key schema. `keys.sk_identifier` is `None` for a table with a
/// simple primary key (partition key only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimaryIndex {
    pub keys: KeySchema,
}

/// A Global Secondary Index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalSecondaryIndex {
    pub name: String,
    pub keys: KeySchema,
}

/// A Local Secondary Index. Always shares the base table's partition key, so only its own
/// sort key attribute name needs to be specified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalSecondaryIndex {
    pub name: String,
    pub sk_identifier: String,
}

/// A condition on an index's sort key, used by [`AwsDynamoDbService::query_items`].
///
/// Mirrors the operators DynamoDB's `KeyConditionExpression` supports for a sort key. Not
/// every variant is implemented by every [`AwsDynamoDbService`] backend yet — [`Self::Prefix`]
/// and [`Self::Between`] are; the others return an `Err` until they're wired up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SortKeyCondition {
    /// `sk = value`
    Equals(String),
    /// `sk < value`
    LessThan(String),
    /// `sk <= value`
    LessThanOrEqual(String),
    /// `sk > value`
    GreaterThan(String),
    /// `sk >= value`
    GreaterThanOrEqual(String),
    /// `sk BETWEEN low AND high`
    Between(String, String),
    /// `begins_with(sk, prefix)`
    Prefix(String),
}

/// Any index that [`AwsDynamoDbService::query_items`] can target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnyIndex {
    Primary(PrimaryIndex),
    Gsi(GlobalSecondaryIndex),
    Lsi(LocalSecondaryIndex),
}

impl From<PrimaryIndex> for AnyIndex {
    fn from(index: PrimaryIndex) -> Self {
        AnyIndex::Primary(index)
    }
}

impl From<GlobalSecondaryIndex> for AnyIndex {
    fn from(index: GlobalSecondaryIndex) -> Self {
        AnyIndex::Gsi(index)
    }
}

impl From<LocalSecondaryIndex> for AnyIndex {
    fn from(index: LocalSecondaryIndex) -> Self {
        AnyIndex::Lsi(index)
    }
}

impl AnyIndex {
    /// DynamoDB index name to pass to Query, or `None` to query the base table.
    pub fn index_name(&self) -> Option<&str> {
        match self {
            AnyIndex::Primary(_) => None,
            AnyIndex::Gsi(index) => Some(&index.name),
            AnyIndex::Lsi(index) => Some(&index.name),
        }
    }

    /// Attribute name of this index's partition key. LSIs always share the base table's
    /// partition key, so `primary_index` is needed to resolve it in that case.
    pub fn pk_identifier<'a>(&'a self, primary_index: &'a PrimaryIndex) -> &'a str {
        match self {
            AnyIndex::Primary(index) => &index.keys.pk_identifier,
            AnyIndex::Gsi(index) => &index.keys.pk_identifier,
            AnyIndex::Lsi(_) => &primary_index.keys.pk_identifier,
        }
    }

    /// Attribute name of this index's sort key, if it has one. Always `Some` for LSIs; the
    /// primary index and GSIs may have been defined without one.
    pub fn sk_identifier(&self) -> Option<&str> {
        match self {
            AnyIndex::Primary(index) => index.keys.sk_identifier.as_deref(),
            AnyIndex::Gsi(index) => index.keys.sk_identifier.as_deref(),
            AnyIndex::Lsi(index) => Some(&index.sk_identifier),
        }
    }
}

/// The stored representation of an item, minus its primary key.
///
/// The primary key attributes are written and read separately by [`AwsDynamoDbService`]
/// implementations (under whatever attribute names the configured [`PrimaryIndex`] specifies),
/// so they deliberately aren't fields on `Container` — that would hardcode fixed attribute
/// names into every stored item, in tension with `PrimaryIndex` being configurable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Container<T> {
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
    /// Reads the [`Container`] stored under the base table's primary key `(pk, sk)` and, if
    /// present, returns the stored `data_version` (for optimistic concurrency control) together
    /// with the deserialized payload. `sk` must be `Some` if and only if the implementation's
    /// configured [`PrimaryIndex`] has a sort key; a mismatch returns an `Err`.
    async fn get_item<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
    ) -> Result<Option<(u64, T)>, String>
    where
        T: serde::de::DeserializeOwned + Send;

    /// DynamoDB put_item() operation without optimistic concurrency control.
    ///
    /// Unconditionally writes `payload` under `(pk, sk)` in `table`, overwriting whatever item
    /// is currently stored (if any) and ignoring any existing `data_version`. The stored
    /// `data_version` is reset to `1`.
    ///
    /// Use this only for writes where last-writer-wins is acceptable; prefer
    /// [`AwsDynamoDbService::put_item`] when concurrent writers must not clobber each other.
    /// `sk` must be `Some` if and only if the implementation's configured [`PrimaryIndex`] has
    /// a sort key; a mismatch returns an `Err`.
    async fn put_item_unconditional<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
        payload: &T,
    ) -> Result<(), String>
    where
        T: Serialize + Send + Sync;

    /// DynamoDB put_item() operation with optimistic concurrency control.
    ///
    /// Writes `payload` under `(pk, sk)` in `table`, but only if the currently stored
    /// `data_version` matches `expected_version` (which must be > 0). Does not accept `0`
    /// (returns an error if passed).
    ///
    /// On success the version is incremented and the new `data_version` is returned. If the
    /// version check fails (a concurrent writer modified or created the item), an `Err` is
    /// returned and the table is left untouched. `sk` must be `Some` if and only if the
    /// implementation's configured [`PrimaryIndex`] has a sort key; a mismatch returns an
    /// `Err`.
    async fn put_item<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
        expected_version: u64,
        payload: &T,
    ) -> Result<u64, String>
    where
        T: Serialize + Send + Sync;

    /// DynamoDB Query operation, against the base table or any
    /// [`GlobalSecondaryIndex`]/[`LocalSecondaryIndex`].
    ///
    /// Returns every [`Container`] stored under partition key `pk` in `index`, narrowed by
    /// `sk_condition` if `Some` (see [`SortKeyCondition`]), or every item under `pk` when
    /// `sk_condition` is `None`. For each matching item the stored `data_version` (for
    /// optimistic concurrency control) is returned together with the deserialized payload. An
    /// empty `Vec` is returned when nothing matches. Returns an `Err` if `sk_condition` is
    /// `Some` but `index` has no sort key, or if the given [`SortKeyCondition`] variant isn't
    /// implemented by this backend.
    async fn query_items<T>(
        &self,
        index: impl Into<AnyIndex> + Send,
        pk: String,
        sk_condition: Option<SortKeyCondition>,
        table: &str,
    ) -> Result<Vec<(u64, T)>, String>
    where
        T: serde::de::DeserializeOwned + Send;

    /// DynamoDB delete_item() operation.
    ///
    /// Removes the [`Container`] stored under `(pk, sk)` in `table`. The operation is
    /// idempotent: deleting a key that does not exist succeeds and is treated as a no-op
    /// (mirroring DynamoDB's DeleteItem semantics). On success `Ok(())` is returned; transport
    /// or service errors are surfaced as an `Err` describing the failure. `sk` must be `Some`
    /// if and only if the implementation's configured [`PrimaryIndex`] has a sort key; a
    /// mismatch returns an `Err`.
    async fn delete_item(&self, pk: String, sk: Option<String>, table: &str) -> Result<(), String>;
}

#[cfg(test)]
mod tests {
    use serde::Serialize;
    use std::collections::HashMap;

    use crate::Container;

    #[test]
    fn test_container_fields_not_renamed() {
        #[derive(Serialize)]
        struct TestPayload {
            magic: u32,
        }

        let container = Container {
            data_version: 5,
            payload: TestPayload { magic: 42 },
        };

        let map: HashMap<String, serde_json::Value> =
            serde_json::from_value(serde_json::to_value(&container).unwrap()).unwrap();

        assert!(map.contains_key("data_version"));
        assert!(map.contains_key("magic"));
    }
}
