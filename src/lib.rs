use std::fmt;

use serde::{Deserialize, Serialize};

#[cfg(feature = "aws")]
pub mod aws;

#[cfg(feature = "mock")]
pub mod mock;

// Only compiled when there is a backend to run it against.
#[cfg(all(test, any(feature = "mock", feature = "dynamodb-local-tests")))]
mod conformance;

/// Everything an [`AwsDynamoDbService`] operation can fail with.
///
/// The distinction that matters to most callers is [`Error::Conflict`] — a lost race that the
/// caller should retry — versus everything else, which signals a bug or an outage. Before this
/// type existed every failure was a `String` and telling those apart meant substring-matching
/// `"Optimistic concurrency conflict"`.
///
/// Marked `#[non_exhaustive]`: match with a wildcard arm, so that new variants (starting with
/// the "already exists" outcome a future `create_item` needs) don't break you.
///
/// The variants that carry a `String` have already had their context formatted in. Backends
/// erase their underlying error types into that string rather than keeping them as a
/// [`std::error::Error::source`], so no `source` is available to walk.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Error {
    /// [`AwsDynamoDbService::create_item`] found an item already stored under the key, so
    /// nothing was written.
    ///
    /// Distinct from [`Error::Conflict`] on purpose: a caller bootstrapping an item wants to
    /// tell "someone else already created this" apart from "my compare-and-swap lost", because
    /// the recovery differs — re-read and merge, versus retry the swap.
    AlreadyExists { pk: String, sk: Option<String> },

    /// An optimistic-concurrency check failed: the item's stored `data_version` was not
    /// `expected_version`, so a concurrent writer got there first and nothing was written.
    ///
    /// Also what [`AwsDynamoDbService::put_item`] returns for an item that does not exist,
    /// since its condition compares an attribute of an existing item and there is no way to
    /// tell the two cases apart from the backend's response. Use
    /// [`AwsDynamoDbService::create_item`] when the item may be absent.
    ///
    /// Deliberately does not report the version actually found: DynamoDB's failed conditional
    /// check doesn't reveal it, so a field for it could not be filled in consistently across
    /// backends.
    Conflict {
        pk: String,
        sk: Option<String>,
        expected_version: u64,
    },

    /// The call was rejected before it reached the backend, because the arguments contradict
    /// the configured schema or the operation's contract — an `sk` that disagrees with the
    /// [`PrimaryIndex`], an `expected_version` of `0`, a sort key condition against an index
    /// that has no sort key.
    ///
    /// Always a caller bug: retrying unchanged will fail the same way.
    InvalidRequest(String),

    /// The operation is meaningful but this backend hasn't implemented it yet — currently the
    /// [`SortKeyCondition`] variants that aren't wired up.
    Unsupported(String),

    /// A payload could not be serialized into, or deserialized out of, its stored
    /// representation. Usually means the stored item no longer matches the `T` being read.
    Serialization(String),

    /// The backend itself failed: transport error, throttling, an unexpected service response.
    /// The one variant that may be worth retrying blindly.
    Service(String),
}

impl Error {
    /// Rejects a counter attribute that would collide with something the backend manages.
    ///
    /// Shared so the two backends can't drift on a check they both perform.
    #[cfg(any(feature = "aws", feature = "mock"))]
    pub(crate) fn reserved_counter_attribute(attribute: &str, reason: &str) -> Self {
        Error::InvalidRequest(format!(
            "add_to_counter() cannot target the attribute {attribute:?}: {reason}"
        ))
    }

    /// Rejects a [`SortKeyCondition`] against an index that has no sort key.
    ///
    /// Shared so the two backends can't drift on the wording of a check they both perform.
    #[cfg(any(feature = "aws", feature = "mock"))]
    pub(crate) fn sk_condition_without_sort_key() -> Self {
        Error::InvalidRequest(
            "a sort key condition was provided but the index has no sort key".to_string(),
        )
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Formatted here rather than at each call site so every backend words a conflict
            // identically.
            Error::AlreadyExists { pk, sk } => {
                write!(f, "Item already exists for pk={pk}, sk={sk:?}")
            }
            Error::Conflict {
                pk,
                sk,
                expected_version,
            } => write!(
                f,
                "Optimistic concurrency conflict for pk={pk}, sk={sk:?}: expected version {expected_version}"
            ),
            Error::InvalidRequest(message)
            | Error::Unsupported(message)
            | Error::Serialization(message)
            | Error::Service(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {}

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

/// Attribute name [`Container::data_version`] serializes to. Must match the field name.
#[cfg(any(feature = "aws", feature = "mock"))]
pub(crate) const DATA_VERSION_ATTRIBUTE: &str = "data_version";

#[cfg(any(feature = "aws", feature = "mock"))]
impl PrimaryIndex {
    /// Rejects a counter attribute the backend can't atomically add to: a primary key attribute
    /// (DynamoDB forbids updating those) or the `data_version` attribute (which
    /// [`AwsDynamoDbService::add_to_counter`] bumps itself, and two updates can't target one
    /// path).
    pub(crate) fn validate_counter_attribute(&self, attribute: &str) -> Result<(), Error> {
        if attribute == self.keys.pk_identifier {
            return Err(Error::reserved_counter_attribute(
                attribute,
                "it is the table's partition key",
            ));
        }
        if self.keys.sk_identifier.as_deref() == Some(attribute) {
            return Err(Error::reserved_counter_attribute(
                attribute,
                "it is the table's sort key",
            ));
        }
        if attribute == DATA_VERSION_ATTRIBUTE {
            return Err(Error::reserved_counter_attribute(
                attribute,
                "it holds the optimistic-concurrency version",
            ));
        }
        Ok(())
    }

    /// Resolves a caller-supplied `sk` against this schema, returning the sort key's attribute
    /// name paired with the value when the table has one.
    ///
    /// [`Error::InvalidRequest`] if `sk` doesn't match whether the table actually has a sort
    /// key. Every [`AwsDynamoDbService`] implementation routes its key handling through here so
    /// the backends agree on what a well-formed key is.
    pub(crate) fn resolve_sk<'a>(
        &'a self,
        sk: &Option<String>,
    ) -> Result<Option<(&'a str, String)>, Error> {
        match (self.keys.sk_identifier.as_deref(), sk) {
            (Some(sk_identifier), Some(sk)) => Ok(Some((sk_identifier, sk.clone()))),
            (None, None) => Ok(None),
            (Some(_), None) => Err(Error::InvalidRequest(
                "PrimaryIndex has a sort key, but none was provided".to_string(),
            )),
            (None, Some(_)) => Err(Error::InvalidRequest(
                "a sort key was provided, but PrimaryIndex has none".to_string(),
            )),
        }
    }
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

/// The `data_version` a newly created item starts at.
///
/// `0` is below it, and means "stored before versioning existed" (see
/// [`Container::data_version`]). Passing `0` to [`AwsDynamoDbService::put_item`] upgrades such a
/// record onto the versioning scheme, landing it here.
pub const INITIAL_DATA_VERSION: u64 = 1;

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
    /// with the deserialized payload.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRequest`] if `sk` is not `Some` exactly when the implementation's
    /// configured [`PrimaryIndex`] has a sort key, [`Error::Serialization`] if the stored item
    /// doesn't deserialize into `T`, [`Error::Service`] if the backend fails.
    async fn get_item<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
    ) -> Result<Option<(u64, T)>, Error>
    where
        T: serde::de::DeserializeOwned + Send;

    /// DynamoDB put_item() operation conditioned on the item not existing.
    ///
    /// Writes `payload` under `(pk, sk)` in `table` at `data_version` 1, but only if no item is
    /// stored under that key — DynamoDB's `attribute_not_exists` idiom. Returns the new
    /// `data_version`, so the caller can carry on with [`AwsDynamoDbService::put_item`] without
    /// re-reading.
    ///
    /// This is the safe way to bootstrap an item that may not exist yet. Branching on
    /// [`AwsDynamoDbService::get_item`] returning `None` and then writing with
    /// [`AwsDynamoDbService::put_item_unconditional`] is *not* safe: two callers can both
    /// observe the item as absent, and the slower one's write silently destroys everything the
    /// faster one committed. Written as a loop, with both failure modes retried:
    ///
    /// ```ignore
    /// loop {
    ///     let result = match db.get_item(pk.clone(), sk.clone(), table).await? {
    ///         Some((version, value)) => {
    ///             db.put_item(pk.clone(), sk.clone(), table, version, &next(value)).await.map(|_| ())
    ///         }
    ///         None => {
    ///             db.create_item(pk.clone(), sk.clone(), table, &initial()).await.map(|_| ())
    ///         }
    ///     };
    ///     match result {
    ///         Ok(()) => break,
    ///         // Someone else got there first, in either branch. Re-read and try again.
    ///         Err(Error::Conflict { .. } | Error::AlreadyExists { .. }) => continue,
    ///         Err(other) => return Err(other),
    ///     }
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::AlreadyExists`] if an item is already stored under the key, leaving it
    /// untouched. This includes a legacy record stored without a `data_version` — the condition
    /// tests the partition key attribute, not the version.
    ///
    /// [`Error::InvalidRequest`] if `sk` is not `Some` exactly when the implementation's
    /// configured [`PrimaryIndex`] has a sort key, [`Error::Serialization`] if `payload` doesn't
    /// serialize, [`Error::Service`] if the backend fails.
    async fn create_item<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
        payload: &T,
    ) -> Result<u64, Error>
    where
        T: Serialize + Send + Sync;

    /// Last-writer-wins write, with no expectation about the item's current contents.
    ///
    /// Writes `payload` under `(pk, sk)` in `table`, overwriting whatever item is currently
    /// stored (if any), and returns the new `data_version`.
    ///
    /// The version *advances*: one past whatever was stored, or [`INITIAL_DATA_VERSION`] if the
    /// key was absent. It is never rewound, so a `data_version` a caller has already spent can
    /// never become valid again. Achieving that costs a read — implementations are not required
    /// to do this in a single round trip, and the AWS backend does not.
    ///
    /// # Deprecated
    ///
    /// Prefer [`AwsDynamoDbService::create_item`] to bootstrap an item and
    /// [`AwsDynamoDbService::put_item`] to update one — including to upgrade a record stored
    /// without a `data_version`, which `put_item` handles by taking `expected_version` of `0`.
    /// Between them they cover every case safely, and both tell the caller when they lost a
    /// race. This operation, by design, cannot.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRequest`] if `sk` is not `Some` exactly when the implementation's
    /// configured [`PrimaryIndex`] has a sort key, [`Error::Serialization`] if `payload` doesn't
    /// serialize, [`Error::Service`] if the backend fails.
    ///
    /// [`Error::Conflict`] if sustained contention on the key exhausts the implementation's
    /// retry budget. Nothing is written in that case; the operation is safe to retry.
    #[deprecated(
        since = "0.3.0",
        note = "use create_item() to bootstrap an item and put_item() to update one; both \
                report a lost race, which this operation cannot"
    )]
    async fn put_item_unconditional<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
        payload: &T,
    ) -> Result<u64, Error>
    where
        T: Serialize + Send + Sync;

    /// DynamoDB put_item() operation with optimistic concurrency control.
    ///
    /// Writes `payload` under `(pk, sk)` in `table`, but only if the currently stored
    /// `data_version` matches `expected_version`.
    ///
    /// On success the version is incremented and the new `data_version` is returned.
    ///
    /// # Upgrading a legacy record
    ///
    /// `expected_version` of `0` means "the item exists but carries no `data_version`" — a
    /// record written before versioning existed, which [`AwsDynamoDbService::get_item`] reports
    /// as version `0`. The write succeeds only if that is still true, and lands the record at
    /// [`INITIAL_DATA_VERSION`], after which it takes part in optimistic concurrency like any
    /// other item.
    ///
    /// This is a compare-and-swap like any other, so the read-modify-write loop needs no special
    /// case: feed back whatever version `get_item` returned, `0` included.
    ///
    /// Note that `0` does *not* mean "create if absent" — use
    /// [`AwsDynamoDbService::create_item`] for that. The two conditions are mutually exclusive:
    /// one requires the item to exist, the other requires it not to.
    ///
    /// # Errors
    ///
    /// [`Error::Conflict`] if the version check fails, leaving the table untouched. Note that
    /// this covers two cases the backend cannot distinguish: a concurrent writer changed the
    /// item, *or* the item does not exist at all — every condition this operation can emit
    /// requires an existing item, so it can never create one.
    ///
    /// [`Error::InvalidRequest`] if `sk` is not `Some` exactly when the implementation's
    /// configured [`PrimaryIndex`] has a sort key. [`Error::Serialization`] if `payload` doesn't
    /// serialize, [`Error::Service`] if the backend fails.
    async fn put_item<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
        expected_version: u64,
        payload: &T,
    ) -> Result<u64, Error>
    where
        T: Serialize + Send + Sync;

    /// DynamoDB UpdateItem `ADD`: atomically adds `delta` to a numeric attribute.
    ///
    /// Returns the attribute's new value. The add happens server-side in a single request, so
    /// concurrent callers each get a distinct result and none of them can lose an increment —
    /// which is what makes this suitable for allocating identifiers. Unlike a
    /// [`AwsDynamoDbService::get_item`] → [`AwsDynamoDbService::put_item`] loop it never
    /// conflicts, so it costs one round trip regardless of contention.
    ///
    /// Creates the item if the key is absent, treating the counter as having been `0`; a first
    /// call with `delta` of `1` therefore returns `1`. `delta` may be negative.
    ///
    /// The item's `data_version` is bumped by the same request, so a counter item stays
    /// versioned like any other and an unversioned one is upgraded onto the scheme by its first
    /// increment. Every other attribute is left alone: this is an update, not the wholesale
    /// replacement that [`AwsDynamoDbService::put_item`] performs, so a counter can live on an
    /// item that also carries a payload.
    ///
    /// Note the trade-off against `put_item`: an `ADD` cannot be conditioned on the value it is
    /// changing, so a caller that must *decide* based on the current value — clamp at a limit,
    /// reject an overdraft — still needs the compare-and-swap loop.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRequest`] if `counter_attribute` is a primary key attribute or
    /// `data_version`, or if `sk` is not `Some` exactly when the implementation's configured
    /// [`PrimaryIndex`] has a sort key.
    ///
    /// [`Error::Service`] if the backend fails — including when the stored attribute exists but
    /// isn't a number, which DynamoDB rejects as a malformed update rather than a conflict.
    /// Never [`Error::Conflict`]: there is no condition to lose.
    async fn add_to_counter(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
        counter_attribute: &str,
        delta: i64,
    ) -> Result<i64, Error>;

    /// DynamoDB Query operation, against the base table or any
    /// [`GlobalSecondaryIndex`]/[`LocalSecondaryIndex`].
    ///
    /// Returns every [`Container`] stored under partition key `pk` in `index`, narrowed by
    /// `sk_condition` if `Some` (see [`SortKeyCondition`]), or every item under `pk` when
    /// `sk_condition` is `None`. For each matching item the stored `data_version` (for
    /// optimistic concurrency control) is returned together with the deserialized payload. An
    /// empty `Vec` is returned when nothing matches.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRequest`] if `sk_condition` is `Some` but `index` has no sort key,
    /// [`Error::Unsupported`] if the given [`SortKeyCondition`] variant isn't implemented by
    /// this backend, [`Error::Serialization`] if a matched item doesn't deserialize into `T`,
    /// [`Error::Service`] if the backend fails.
    async fn query_items<T>(
        &self,
        index: impl Into<AnyIndex> + Send,
        pk: String,
        sk_condition: Option<SortKeyCondition>,
        table: &str,
    ) -> Result<Vec<(u64, T)>, Error>
    where
        T: serde::de::DeserializeOwned + Send;

    /// DynamoDB delete_item() operation.
    ///
    /// Removes the [`Container`] stored under `(pk, sk)` in `table`. The operation is
    /// idempotent: deleting a key that does not exist succeeds and is treated as a no-op
    /// (mirroring DynamoDB's DeleteItem semantics).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRequest`] if `sk` is not `Some` exactly when the implementation's
    /// configured [`PrimaryIndex`] has a sort key, [`Error::Service`] if the backend fails.
    async fn delete_item(&self, pk: String, sk: Option<String>, table: &str) -> Result<(), Error>;
}

#[cfg(test)]
mod tests {
    use serde::Serialize;
    use std::collections::HashMap;

    use crate::{Container, Error};

    #[test]
    fn conflict_display_names_the_key_and_expected_version() {
        let err = Error::Conflict {
            pk: "device#1".to_string(),
            sk: Some("METADATA".to_string()),
            expected_version: 4,
        };

        assert_eq!(
            err.to_string(),
            r#"Optimistic concurrency conflict for pk=device#1, sk=Some("METADATA"): expected version 4"#
        );
    }

    #[test]
    fn error_is_a_std_error() {
        // Callers need `?` into Box<dyn Error> / anyhow to keep working.
        fn boxed() -> Result<(), Box<dyn std::error::Error>> {
            Err(Error::Service("backend exploded".to_string()))?;
            Ok(())
        }

        assert_eq!(boxed().unwrap_err().to_string(), "backend exploded");
    }

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
