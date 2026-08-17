//! Backend-agnostic conformance suite for [`AwsDynamoDbService`].
//!
//! Every test here is a generic `async fn` over a [`Harness`], so the exact same assertions run
//! against every implementation of the trait. [`MockDynamoDb`](crate::mock::MockDynamoDb) runs
//! them on `cargo test`; a real DynamoDB Local instance runs them behind the
//! `dynamodb-local-tests` feature (see [`local`]).
//!
//! This exists because the mock is only useful insofar as it behaves like the backend it stands
//! in for, and "remember to update the mock too" had already failed once: the mock accepted keys
//! that the AWS backend rejects. A shared suite makes that class of drift a test failure.
//!
//! # Adding a test
//!
//! Write the generic `async fn` here, then add its name to the list inside
//! [`conformance_suite!`]. Both backends pick it up automatically.
//!
//! Tests must reach the store only through [`AwsDynamoDbService`] — helpers like
//! [`MockDynamoDb::insert`](crate::mock::MockDynamoDb::insert) have no counterpart on a real
//! table. Use [`seed`] to reach a specific `data_version`.

use serde::{Deserialize, Serialize};

use crate::{
    AwsDynamoDbService, Error, GlobalSecondaryIndex, INITIAL_DATA_VERSION, KeySchema,
    LocalSecondaryIndex, PrimaryIndex, SortKeyCondition,
};

#[cfg(feature = "dynamodb-local-tests")]
pub(crate) mod local;
#[cfg(feature = "mock")]
pub(crate) mod mock;

/// A backend under test, plus the tables the suite is allowed to touch.
///
/// `table` and `other_table` are distinct and both start empty. `other_table` exists so the
/// suite can prove that operations don't leak across tables.
pub(crate) struct Fixture<D> {
    pub db: D,
    pub table: String,
    pub other_table: String,
}

/// Constructs backends for the conformance suite.
///
/// Implementations must hand back a store that is empty and isolated from every other fixture,
/// so tests can run concurrently and in any order.
pub(crate) trait Harness {
    type Db: AwsDynamoDbService;

    /// Returns a fixture whose tables have `keys` as their primary key schema, and which carry
    /// the [`gsi1`] and (when `keys` has a sort key) [`lsi1`] secondary indexes.
    async fn setup(&self, keys: KeySchema) -> Fixture<Self::Db>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Payload {
    name: String,
    count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct PayloadWithGsi {
    gsi1pk: String,
    gsi1sk: String,
    name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct PayloadWithLsi {
    lsi1sk: String,
    name: String,
}

/// A composite primary key. The attribute names are arbitrary — nothing in either backend
/// requires the "pk"/"sk" convention, it's just what this suite picked.
pub(crate) fn composite_key() -> KeySchema {
    KeySchema {
        pk_identifier: "pk".to_string(),
        sk_identifier: Some("sk".to_string()),
    }
}

/// A simple primary key: partition key only, no sort key.
pub(crate) fn partition_only_key() -> KeySchema {
    KeySchema {
        pk_identifier: "pk".to_string(),
        sk_identifier: None,
    }
}

/// The GSI every fixture carries. A [`Harness`] backed by real tables must create it.
pub(crate) fn gsi1() -> GlobalSecondaryIndex {
    GlobalSecondaryIndex {
        name: "GSI1".to_string(),
        keys: KeySchema {
            pk_identifier: "gsi1pk".to_string(),
            sk_identifier: Some("gsi1sk".to_string()),
        },
    }
}

/// The LSI fixtures with a composite primary key carry. DynamoDB only allows an LSI on a table
/// that has a sort key, so partition-only fixtures omit it.
pub(crate) fn lsi1() -> LocalSecondaryIndex {
    LocalSecondaryIndex {
        name: "LSI1".to_string(),
        sk_identifier: "lsi1sk".to_string(),
    }
}

/// A GSI with no sort key. Never created on any table — it's only used to check that a sort key
/// condition against a sort-key-less index is rejected before a request is ever sent.
fn gsi_without_sort_key() -> GlobalSecondaryIndex {
    GlobalSecondaryIndex {
        name: "GSI2".to_string(),
        keys: KeySchema {
            pk_identifier: "gsi2pk".to_string(),
            sk_identifier: None,
        },
    }
}

pub(crate) fn primary_index(keys: KeySchema) -> PrimaryIndex {
    PrimaryIndex { keys }
}

fn key(pk: &str, sk: Option<&str>) -> (String, Option<String>) {
    (pk.to_string(), sk.map(String::from))
}

/// Writes `payload` under `(pk, sk)` and leaves it stored at `data_version == version`.
///
/// Goes through the trait rather than any backend-specific fixture helper, which means it can
/// only reach versions >= 1: it creates the item at version 1 and then CASes it forward. Real
/// tables have no other way in, and a legacy record (version 0) can't be produced this way at
/// all — that's a Phase 4 concern.
async fn seed<D, T>(db: &D, pk: &str, sk: Option<&str>, table: &str, version: u64, payload: &T)
where
    D: AwsDynamoDbService,
    T: Serialize + Send + Sync,
{
    assert!(version >= 1, "seed() cannot produce data_version {version}");

    let (pk, sk) = key(pk, sk);
    db.create_item(pk.clone(), sk.clone(), table, payload)
        .await
        .expect("seed: initial write failed");

    for current in 1..version {
        db.put_item(pk.clone(), sk.clone(), table, current, payload)
            .await
            .expect("seed: version bump failed");
    }
}

// ---------------------------------------------------------------------------------------------
// get_item
// ---------------------------------------------------------------------------------------------

pub(crate) async fn get_item_returns_stored_payload_with_version<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let payload = Payload {
        name: "balu".to_string(),
        count: 3,
    };
    seed(&f.db, "device#1", Some("METADATA"), &f.table, 7, &payload).await;

    let (pk, sk) = key("device#1", Some("METADATA"));
    let result: Option<(u64, Payload)> = f.db.get_item(pk, sk, &f.table).await.unwrap();

    assert_eq!(result, Some((7, payload)));
}

pub(crate) async fn get_item_returns_none_for_missing_key<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;

    let (pk, sk) = key("device#missing", Some("METADATA"));
    let result: Option<(u64, Payload)> = f.db.get_item(pk, sk, &f.table).await.unwrap();

    assert_eq!(result, None);
}

// ---------------------------------------------------------------------------------------------
// create_item
// ---------------------------------------------------------------------------------------------

pub(crate) async fn create_item_creates_item_at_version_one<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let (pk, sk) = key("device#1", Some("METADATA"));
    let payload = Payload {
        name: "balu".to_string(),
        count: 1,
    };

    let version =
        f.db.create_item(pk.clone(), sk.clone(), &f.table, &payload)
            .await
            .unwrap();

    assert_eq!(version, 1);
    let result: Option<(u64, Payload)> =
        f.db.get_item(pk.clone(), sk.clone(), &f.table)
            .await
            .unwrap();
    assert_eq!(result, Some((1, payload.clone())));

    // The returned version is a usable CAS token, so a caller can carry on without re-reading.
    let next =
        f.db.put_item(pk, sk, &f.table, version, &payload)
            .await
            .unwrap();
    assert_eq!(next, 2);
}

pub(crate) async fn create_item_rejects_an_existing_item<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let (pk, sk) = key("device#1", Some("METADATA"));
    let original = Payload {
        name: "original".to_string(),
        count: 1,
    };
    f.db.create_item(pk.clone(), sk.clone(), &f.table, &original)
        .await
        .unwrap();

    let err =
        f.db.create_item(
            pk.clone(),
            sk.clone(),
            &f.table,
            &Payload {
                name: "clobber".to_string(),
                count: 99,
            },
        )
        .await
        .unwrap_err();

    assert!(matches!(err, Error::AlreadyExists { .. }), "{err:?}");
    let result: Option<(u64, Payload)> = f.db.get_item(pk, sk, &f.table).await.unwrap();
    assert_eq!(
        result,
        Some((1, original)),
        "a refused create must leave the stored item untouched"
    );
}

/// The condition tests whether the item exists, not what version it holds, so an item that has
/// been updated well past version 1 still blocks a create.
pub(crate) async fn create_item_rejects_an_item_at_any_version<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let (pk, sk) = key("device#1", Some("METADATA"));
    let original = Payload {
        name: "original".to_string(),
        count: 1,
    };
    seed(&f.db, "device#1", Some("METADATA"), &f.table, 5, &original).await;

    let err =
        f.db.create_item(
            pk.clone(),
            sk.clone(),
            &f.table,
            &Payload {
                name: "clobber".to_string(),
                count: 99,
            },
        )
        .await
        .unwrap_err();

    assert!(matches!(err, Error::AlreadyExists { .. }), "{err:?}");
    let result: Option<(u64, Payload)> = f.db.get_item(pk, sk, &f.table).await.unwrap();
    assert_eq!(result, Some((5, original)));
}

/// A deleted key is absent again, so it can be recreated.
pub(crate) async fn create_item_succeeds_again_after_delete<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let (pk, sk) = key("device#1", Some("METADATA"));
    let payload = Payload {
        name: "balu".to_string(),
        count: 1,
    };
    f.db.create_item(pk.clone(), sk.clone(), &f.table, &payload)
        .await
        .unwrap();
    f.db.delete_item(pk.clone(), sk.clone(), &f.table)
        .await
        .unwrap();

    let version = f.db.create_item(pk, sk, &f.table, &payload).await.unwrap();

    assert_eq!(version, 1);
}

// ---------------------------------------------------------------------------------------------
// put_item (optimistic concurrency control)
// ---------------------------------------------------------------------------------------------

pub(crate) async fn put_item_updates_when_version_matches<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let (pk, sk) = key("device#1", Some("METADATA"));
    seed(
        &f.db,
        "device#1",
        Some("METADATA"),
        &f.table,
        5,
        &Payload {
            name: "old".to_string(),
            count: 1,
        },
    )
    .await;
    let updated = Payload {
        name: "new".to_string(),
        count: 2,
    };

    let new_version =
        f.db.put_item(pk.clone(), sk.clone(), &f.table, 5, &updated)
            .await
            .unwrap();

    assert_eq!(new_version, 6);
    let result: Option<(u64, Payload)> = f.db.get_item(pk, sk, &f.table).await.unwrap();
    assert_eq!(result, Some((6, updated)));
}

pub(crate) async fn put_item_rejects_stale_version<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let (pk, sk) = key("device#1", Some("METADATA"));
    let original = Payload {
        name: "old".to_string(),
        count: 1,
    };
    seed(&f.db, "device#1", Some("METADATA"), &f.table, 5, &original).await;

    let err =
        f.db.put_item(
            pk.clone(),
            sk.clone(),
            &f.table,
            4,
            &Payload {
                name: "new".to_string(),
                count: 2,
            },
        )
        .await
        .unwrap_err();

    assert!(
        matches!(
            err,
            Error::Conflict {
                expected_version: 4,
                ..
            }
        ),
        "{err:?}"
    );
    // Display is formatted by Error itself rather than at each call site, so asserting the exact
    // text here — in a test both backends run — is what keeps their wording from diverging.
    assert_eq!(
        err.to_string(),
        r#"Optimistic concurrency conflict for pk=device#1, sk=Some("METADATA"): expected version 4"#
    );
    // The stored item must be left untouched after a rejected write.
    let result: Option<(u64, Payload)> = f.db.get_item(pk, sk, &f.table).await.unwrap();
    assert_eq!(result, Some((5, original)));
}

pub(crate) async fn put_item_rejects_expected_version_zero<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let (pk, sk) = key("device#1", Some("METADATA"));
    let payload = Payload {
        name: "balu".to_string(),
        count: 1,
    };
    seed(&f.db, "device#1", Some("METADATA"), &f.table, 1, &payload).await;

    let err =
        f.db.put_item(pk, sk, &f.table, 0, &payload)
            .await
            .unwrap_err();

    assert!(matches!(err, Error::InvalidRequest(_)), "{err:?}");
}

/// `put_item`'s condition compares an attribute of an existing item, so against a missing key it
/// always fails: there is no way to express "create, but only if absent" through this operation.
/// The gap `create_item` is meant to close.
pub(crate) async fn put_item_cannot_create_a_missing_item<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let (pk, sk) = key("device#missing", Some("METADATA"));

    let err =
        f.db.put_item(
            pk.clone(),
            sk.clone(),
            &f.table,
            1,
            &Payload {
                name: "balu".to_string(),
                count: 1,
            },
        )
        .await
        .unwrap_err();

    // Indistinguishable from a genuine lost race: the backend reports the same failed
    // condition either way, which is why `create_item` has to be its own operation.
    assert!(matches!(err, Error::Conflict { .. }), "{err:?}");
    let result: Option<(u64, Payload)> = f.db.get_item(pk, sk, &f.table).await.unwrap();
    assert_eq!(result, None, "a rejected write must not create the item");
}

// ---------------------------------------------------------------------------------------------
// put_item_unconditional
// ---------------------------------------------------------------------------------------------

#[allow(
    deprecated,
    reason = "exercising the deprecated operation is the point of this test"
)]
pub(crate) async fn put_item_unconditional_creates_item_with_version_one<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let (pk, sk) = key("device#1", Some("METADATA"));
    let payload = Payload {
        name: "balu".to_string(),
        count: 1,
    };

    let version =
        f.db.put_item_unconditional(pk.clone(), sk.clone(), &f.table, &payload)
            .await
            .unwrap();

    assert_eq!(version, INITIAL_DATA_VERSION);
    let result: Option<(u64, Payload)> = f.db.get_item(pk, sk, &f.table).await.unwrap();
    assert_eq!(result, Some((1, payload)));
}

#[allow(
    deprecated,
    reason = "exercising the deprecated operation is the point of this test"
)]
pub(crate) async fn put_item_unconditional_overwrites_and_advances_version<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let (pk, sk) = key("device#1", Some("METADATA"));
    seed(
        &f.db,
        "device#1",
        Some("METADATA"),
        &f.table,
        4,
        &Payload {
            name: "old".to_string(),
            count: 1,
        },
    )
    .await;
    let updated = Payload {
        name: "new".to_string(),
        count: 2,
    };

    let version =
        f.db.put_item_unconditional(pk.clone(), sk.clone(), &f.table, &updated)
            .await
            .unwrap();

    // The payload is replaced wholesale, but the version moves forward from what was stored
    // rather than being reset.
    assert_eq!(version, 5);
    let result: Option<(u64, Payload)> = f.db.get_item(pk, sk, &f.table).await.unwrap();
    assert_eq!(result, Some((5, updated)));
}

/// Repeated unconditional writes must keep climbing, never plateau or reset.
#[allow(
    deprecated,
    reason = "exercising the deprecated operation is the point of this test"
)]
pub(crate) async fn repeated_unconditional_writes_keep_advancing_the_version<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let (pk, sk) = key("device#1", Some("METADATA"));
    let payload = Payload {
        name: "balu".to_string(),
        count: 1,
    };

    for expected in 1..=4 {
        let version =
            f.db.put_item_unconditional(pk.clone(), sk.clone(), &f.table, &payload)
                .await
                .unwrap();
        assert_eq!(version, expected);
    }

    let result: Option<(u64, Payload)> = f.db.get_item(pk, sk, &f.table).await.unwrap();
    assert_eq!(result, Some((4, payload)));
}

// ---------------------------------------------------------------------------------------------
// query_items
// ---------------------------------------------------------------------------------------------

pub(crate) async fn query_items_returns_matching_items_sorted_by_sort_key<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let first = Payload {
        name: "first".to_string(),
        count: 1,
    };
    let second = Payload {
        name: "second".to_string(),
        count: 2,
    };
    // Written out of sort-key order to verify the result comes back sorted.
    seed(&f.db, "device#1", Some("EVENT#2"), &f.table, 4, &second).await;
    seed(&f.db, "device#1", Some("EVENT#1"), &f.table, 3, &first).await;

    let result: Vec<(u64, Payload)> =
        f.db.query_items(
            primary_index(composite_key()),
            "device#1".to_string(),
            Some(SortKeyCondition::Prefix("EVENT#".to_string())),
            &f.table,
        )
        .await
        .unwrap();

    assert_eq!(result, vec![(3, first), (4, second)]);
}

pub(crate) async fn query_items_excludes_non_matching_keys<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let event = Payload {
        name: "event".to_string(),
        count: 1,
    };
    let metadata = Payload {
        name: "metadata".to_string(),
        count: 2,
    };
    seed(&f.db, "device#1", Some("EVENT#1"), &f.table, 1, &event).await;
    // Different sort-key prefix, same partition.
    seed(&f.db, "device#1", Some("METADATA"), &f.table, 1, &metadata).await;
    // Matching sort-key prefix but a different partition.
    seed(&f.db, "device#2", Some("EVENT#1"), &f.table, 1, &event).await;
    // Matching keys but a different table.
    seed(
        &f.db,
        "device#1",
        Some("EVENT#1"),
        &f.other_table,
        1,
        &event,
    )
    .await;

    let result: Vec<(u64, Payload)> =
        f.db.query_items(
            primary_index(composite_key()),
            "device#1".to_string(),
            Some(SortKeyCondition::Prefix("EVENT#".to_string())),
            &f.table,
        )
        .await
        .unwrap();

    assert_eq!(result, vec![(1, event)]);
}

pub(crate) async fn query_items_returns_empty_when_no_match<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;

    let result: Vec<(u64, Payload)> =
        f.db.query_items(
            primary_index(composite_key()),
            "device#1".to_string(),
            Some(SortKeyCondition::Prefix("EVENT#".to_string())),
            &f.table,
        )
        .await
        .unwrap();

    assert!(result.is_empty());
}

pub(crate) async fn query_items_via_between_condition<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
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
    seed(&f.db, "device#1", Some("EVENT#1"), &f.table, 1, &low).await;
    seed(&f.db, "device#1", Some("EVENT#2"), &f.table, 1, &mid).await;
    seed(&f.db, "device#1", Some("EVENT#3"), &f.table, 1, &high).await;

    let result: Vec<(u64, Payload)> =
        f.db.query_items(
            primary_index(composite_key()),
            "device#1".to_string(),
            Some(SortKeyCondition::Between(
                "EVENT#1".to_string(),
                "EVENT#2".to_string(),
            )),
            &f.table,
        )
        .await
        .unwrap();

    assert_eq!(result, vec![(1, low), (1, mid)]);
}

pub(crate) async fn query_items_via_gsi_uses_item_attributes<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
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
    seed(&f.db, "device#1", Some("METADATA"), &f.table, 1, &matching).await;
    seed(
        &f.db,
        "device#2",
        Some("METADATA"),
        &f.table,
        1,
        &other_team,
    )
    .await;
    seed(
        &f.db,
        "device#3",
        Some("METADATA"),
        &f.table,
        1,
        &non_matching_sk,
    )
    .await;

    let result: Vec<(u64, PayloadWithGsi)> =
        f.db.query_items(
            gsi1(),
            "team#a".to_string(),
            Some(SortKeyCondition::Prefix("user#".to_string())),
            &f.table,
        )
        .await
        .unwrap();

    assert_eq!(result, vec![(1, matching)]);
}

pub(crate) async fn query_items_via_lsi_shares_primary_partition_key<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let matching = PayloadWithLsi {
        lsi1sk: "2026-01-01".to_string(),
        name: "matching".to_string(),
    };
    let other_partition = PayloadWithLsi {
        lsi1sk: "2026-01-01".to_string(),
        name: "other partition".to_string(),
    };
    // LSIs share the base table's partition key, so this item must be found by its primary-key
    // pk ("device#1"), not by any attribute inside the payload.
    seed(&f.db, "device#1", Some("METADATA"), &f.table, 1, &matching).await;
    seed(
        &f.db,
        "device#2",
        Some("METADATA"),
        &f.table,
        1,
        &other_partition,
    )
    .await;

    let result: Vec<(u64, PayloadWithLsi)> =
        f.db.query_items(
            lsi1(),
            "device#1".to_string(),
            Some(SortKeyCondition::Prefix("2026".to_string())),
            &f.table,
        )
        .await
        .unwrap();

    assert_eq!(result, vec![(1, matching)]);
}

pub(crate) async fn query_items_rejects_sk_condition_when_index_has_no_sort_key<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;

    let err =
        f.db.query_items::<Payload>(
            gsi_without_sort_key(),
            "device#1".to_string(),
            Some(SortKeyCondition::Prefix("EVENT#".to_string())),
            &f.table,
        )
        .await
        .unwrap_err();

    assert!(matches!(err, Error::InvalidRequest(_)), "{err:?}");
}

pub(crate) async fn query_items_rejects_unimplemented_sort_key_condition<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;

    let err =
        f.db.query_items::<Payload>(
            primary_index(composite_key()),
            "device#1".to_string(),
            Some(SortKeyCondition::Equals("METADATA".to_string())),
            &f.table,
        )
        .await
        .unwrap_err();

    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
}

// ---------------------------------------------------------------------------------------------
// delete_item
// ---------------------------------------------------------------------------------------------

pub(crate) async fn delete_item_removes_existing_item<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let (pk, sk) = key("device#1", Some("METADATA"));
    seed(
        &f.db,
        "device#1",
        Some("METADATA"),
        &f.table,
        1,
        &Payload {
            name: "balu".to_string(),
            count: 1,
        },
    )
    .await;

    f.db.delete_item(pk.clone(), sk.clone(), &f.table)
        .await
        .unwrap();

    let result: Option<(u64, Payload)> = f.db.get_item(pk, sk, &f.table).await.unwrap();
    assert_eq!(result, None);
}

pub(crate) async fn delete_item_is_idempotent_for_missing_key<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let (pk, sk) = key("device#missing", Some("METADATA"));

    // Deleting a key that was never written must succeed as a no-op.
    f.db.delete_item(pk, sk, &f.table).await.unwrap();
}

pub(crate) async fn delete_item_only_removes_targeted_key<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let kept = Payload {
        name: "kept".to_string(),
        count: 1,
    };
    seed(&f.db, "device#1", Some("EVENT#1"), &f.table, 1, &kept).await;
    seed(
        &f.db,
        "device#1",
        Some("EVENT#2"),
        &f.table,
        1,
        &Payload {
            name: "gone".to_string(),
            count: 2,
        },
    )
    .await;

    let (pk, sk) = key("device#1", Some("EVENT#2"));
    f.db.delete_item(pk, sk, &f.table).await.unwrap();

    let (pk, sk) = key("device#1", Some("EVENT#1"));
    let result: Option<(u64, Payload)> = f.db.get_item(pk, sk, &f.table).await.unwrap();
    assert_eq!(result, Some((1, kept)));
}

// ---------------------------------------------------------------------------------------------
// Key schema handling
// ---------------------------------------------------------------------------------------------

pub(crate) async fn partition_only_primary_key_round_trip<H: Harness>(h: &H) {
    let f = h.setup(partition_only_key()).await;
    let payload = Payload {
        name: "balu".to_string(),
        count: 1,
    };

    let version =
        f.db.create_item("device#1".to_string(), None, &f.table, &payload)
            .await
            .unwrap();
    assert_eq!(version, INITIAL_DATA_VERSION);

    let result: Option<(u64, Payload)> =
        f.db.get_item("device#1".to_string(), None, &f.table)
            .await
            .unwrap();
    assert_eq!(result, Some((1, payload)));

    let updated = Payload {
        name: "updated".to_string(),
        count: 2,
    };
    let new_version =
        f.db.put_item("device#1".to_string(), None, &f.table, 1, &updated)
            .await
            .unwrap();
    assert_eq!(new_version, 2);

    f.db.delete_item("device#1".to_string(), None, &f.table)
        .await
        .unwrap();

    let result: Option<(u64, Payload)> =
        f.db.get_item("device#1".to_string(), None, &f.table)
            .await
            .unwrap();
    assert_eq!(result, None);
}

/// Every keyed operation must reject an `sk` argument that disagrees with the configured
/// [`PrimaryIndex`], in both directions.
#[allow(
    deprecated,
    reason = "the deprecated operation must validate keys like the others"
)]
pub(crate) async fn operations_reject_sort_key_not_matching_schema<H: Harness>(h: &H) {
    let payload = Payload {
        name: "balu".to_string(),
        count: 1,
    };

    // Table has a sort key, caller omitted it.
    let f = h.setup(composite_key()).await;
    let missing = |err: Error| {
        assert!(
            matches!(&err, Error::InvalidRequest(message)
                if message.contains("PrimaryIndex has a sort key, but none was provided")),
            "{err:?}"
        )
    };
    missing(
        f.db.get_item::<Payload>("device#1".to_string(), None, &f.table)
            .await
            .unwrap_err(),
    );
    missing(
        f.db.create_item("device#1".to_string(), None, &f.table, &payload)
            .await
            .unwrap_err(),
    );
    missing(
        f.db.put_item_unconditional("device#1".to_string(), None, &f.table, &payload)
            .await
            .unwrap_err(),
    );
    missing(
        f.db.put_item("device#1".to_string(), None, &f.table, 1, &payload)
            .await
            .unwrap_err(),
    );
    missing(
        f.db.delete_item("device#1".to_string(), None, &f.table)
            .await
            .unwrap_err(),
    );

    // Table has no sort key, caller supplied one.
    let f = h.setup(partition_only_key()).await;
    let (pk, sk) = key("device#1", Some("METADATA"));
    let unexpected = |err: Error| {
        assert!(
            matches!(&err, Error::InvalidRequest(message)
                if message.contains("a sort key was provided, but PrimaryIndex has none")),
            "{err:?}"
        )
    };
    unexpected(
        f.db.get_item::<Payload>(pk.clone(), sk.clone(), &f.table)
            .await
            .unwrap_err(),
    );
    unexpected(
        f.db.create_item(pk.clone(), sk.clone(), &f.table, &payload)
            .await
            .unwrap_err(),
    );
    unexpected(
        f.db.put_item_unconditional(pk.clone(), sk.clone(), &f.table, &payload)
            .await
            .unwrap_err(),
    );
    unexpected(
        f.db.put_item(pk.clone(), sk.clone(), &f.table, 1, &payload)
            .await
            .unwrap_err(),
    );
    unexpected(f.db.delete_item(pk, sk, &f.table).await.unwrap_err());
}

/// Two callers both observe an item as absent and race to bootstrap it. The loser must be told,
/// and the winner's committed update must survive.
///
/// The interleaving that used to destroy a committed write, from the v0.2.0 bootstrap-race
/// issue. Replayed sequentially: each operation is individually atomic on both backends, so no
/// real concurrency is needed to hit it — only the ordering.
pub(crate) async fn concurrent_bootstrap_is_rejected_for_the_loser<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let (pk, sk) = key("counter#catalogue", Some("METADATA"));

    // Steps 1 and 2: both actors observe the item as absent.
    let a_seen: Option<(u64, Payload)> =
        f.db.get_item(pk.clone(), sk.clone(), &f.table)
            .await
            .unwrap();
    let b_seen: Option<(u64, Payload)> =
        f.db.get_item(pk.clone(), sk.clone(), &f.table)
            .await
            .unwrap();
    assert_eq!(a_seen, None);
    assert_eq!(b_seen, None);

    // Step 3: actor A takes the `None` branch and bootstraps the item.
    let a_initial = Payload {
        name: "A".to_string(),
        count: 1,
    };
    let a_created =
        f.db.create_item(pk.clone(), sk.clone(), &f.table, &a_initial)
            .await
            .unwrap();
    assert_eq!(a_created, 1);

    // Step 4: actor A commits an update on top of its own bootstrap.
    let a_updated = Payload {
        name: "A".to_string(),
        count: 2,
    };
    let a_version =
        f.db.put_item(pk.clone(), sk.clone(), &f.table, a_created, &a_updated)
            .await
            .unwrap();
    assert_eq!(a_version, 2);

    // Step 5: actor B takes the `None` branch it decided on back at step 2. The item is no
    // longer absent, so the create is refused instead of silently landing.
    let err =
        f.db.create_item(
            pk.clone(),
            sk.clone(),
            &f.table,
            &Payload {
                name: "B".to_string(),
                count: 1,
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::AlreadyExists { .. }), "{err:?}");

    // Actor A's committed update survives, and actor B knows to re-read and retry.
    let stored: Option<(u64, Payload)> = f.db.get_item(pk, sk, &f.table).await.unwrap();
    assert_eq!(stored, Some((2, a_updated)));
}

// ---------------------------------------------------------------------------------------------
// Version monotonicity
// ---------------------------------------------------------------------------------------------

/// A `data_version` that has already been spent must never become valid again.
///
/// `put_item_unconditional` used to reset the version to 1 rather than advancing it, which
/// revived spent compare-and-swap tokens — a textbook ABA. A CAS token is only meaningful if it
/// is monotonic per key.
#[allow(
    deprecated,
    reason = "exercising the deprecated operation is the point of this test"
)]
pub(crate) async fn unconditional_write_does_not_revive_a_stale_version_token<H: Harness>(h: &H) {
    let f = h.setup(composite_key()).await;
    let (pk, sk) = key("device#1", Some("METADATA"));
    let payload = Payload {
        name: "balu".to_string(),
        count: 1,
    };

    // A caller reads version 1 and holds on to it.
    seed(&f.db, "device#1", Some("METADATA"), &f.table, 1, &payload).await;
    let held_version = 1;

    // Someone else advances the item to version 2, spending that token.
    f.db.put_item(pk.clone(), sk.clone(), &f.table, held_version, &payload)
        .await
        .unwrap();
    let err =
        f.db.put_item(pk.clone(), sk.clone(), &f.table, held_version, &payload)
            .await
            .unwrap_err();
    assert!(matches!(err, Error::Conflict { .. }), "{err:?}");

    // An unconditional write advances the version rather than rewinding it...
    let after_unconditional =
        f.db.put_item_unconditional(pk.clone(), sk.clone(), &f.table, &payload)
            .await
            .unwrap();
    assert_eq!(after_unconditional, 3);

    // ...so the spent token stays spent.
    let err =
        f.db.put_item(pk, sk, &f.table, held_version, &payload)
            .await
            .unwrap_err();
    assert!(matches!(err, Error::Conflict { .. }), "{err:?}");
}

/// Generates a `#[tokio::test]` per conformance test, bound to `$harness`.
///
/// Invoke once per backend, from a module of its own:
///
/// ```ignore
/// crate::conformance::conformance_suite!(MyHarness);
/// ```
macro_rules! conformance_suite {
    ($harness:expr) => {
        $crate::conformance::conformance_suite!(@cases $harness;
            get_item_returns_stored_payload_with_version,
            get_item_returns_none_for_missing_key,
            create_item_creates_item_at_version_one,
            create_item_rejects_an_existing_item,
            create_item_rejects_an_item_at_any_version,
            create_item_succeeds_again_after_delete,
            put_item_updates_when_version_matches,
            put_item_rejects_stale_version,
            put_item_rejects_expected_version_zero,
            put_item_cannot_create_a_missing_item,
            put_item_unconditional_creates_item_with_version_one,
            put_item_unconditional_overwrites_and_advances_version,
            repeated_unconditional_writes_keep_advancing_the_version,
            query_items_returns_matching_items_sorted_by_sort_key,
            query_items_excludes_non_matching_keys,
            query_items_returns_empty_when_no_match,
            query_items_via_between_condition,
            query_items_via_gsi_uses_item_attributes,
            query_items_via_lsi_shares_primary_partition_key,
            query_items_rejects_sk_condition_when_index_has_no_sort_key,
            query_items_rejects_unimplemented_sort_key_condition,
            delete_item_removes_existing_item,
            delete_item_is_idempotent_for_missing_key,
            delete_item_only_removes_targeted_key,
            partition_only_primary_key_round_trip,
            operations_reject_sort_key_not_matching_schema,
            concurrent_bootstrap_is_rejected_for_the_loser,
            unconditional_write_does_not_revive_a_stale_version_token,
        );
    };
    (@cases $harness:expr; $($name:ident),* $(,)?) => {
        $(
            #[tokio::test]
            async fn $name() {
                $crate::conformance::$name(&$harness).await;
            }
        )*
    };
}

pub(crate) use conformance_suite;
