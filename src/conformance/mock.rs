//! Runs the conformance suite against [`MockDynamoDb`].
//!
//! These are ordinary unit tests — `cargo test` runs them with no external dependencies.

use crate::KeySchema;
use crate::conformance::{Fixture, Harness, primary_index};
use crate::mock::MockDynamoDb;

/// The mock stores every table in one map keyed by `(table, pk, sk)`, so two table names within
/// a fresh store are already isolated from each other and no setup is needed beyond constructing
/// it. Secondary indexes are resolved from item attributes at query time rather than declared,
/// so [`gsi1`](crate::conformance::gsi1) and [`lsi1`](crate::conformance::lsi1) need no wiring
/// here.
pub(crate) struct MockHarness;

impl Harness for MockHarness {
    type Db = MockDynamoDb;

    async fn setup(&self, keys: KeySchema) -> Fixture<Self::Db> {
        Fixture {
            db: MockDynamoDb::new(primary_index(keys)),
            table: "ConformanceTable".to_string(),
            other_table: "ConformanceOtherTable".to_string(),
        }
    }
}

crate::conformance::conformance_suite!(MockHarness);
