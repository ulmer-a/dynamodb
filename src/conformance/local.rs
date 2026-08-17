//! Runs the conformance suite against a real DynamoDB via [DynamoDB Local].
//!
//! This is the half of the suite that makes the mock's behaviour falsifiable: the mock is only
//! trustworthy if the same assertions pass against an actual DynamoDB engine.
//!
//! Gated behind the `dynamodb-local-tests` feature because it needs a server running. Start one
//! and run the suite with:
//!
//! ```sh
//! docker run --rm -d -p 8000:8000 amazon/dynamodb-local
//! cargo test --features dynamodb-local-tests
//! ```
//!
//! The endpoint defaults to `http://localhost:8000` and can be overridden with
//! `DYNAMODB_LOCAL_ENDPOINT`. Credentials are dummies — DynamoDB Local accepts anything but the
//! SDK insists on something being set.
//!
//! Each fixture creates its own freshly-named tables and leaves them behind; a DynamoDB Local
//! container is disposable, so they're cleaned up by restarting it.
//!
//! [DynamoDB Local]: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/DynamoDBLocal.html

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use aws_sdk_dynamodb::types::{
    AttributeDefinition, BillingMode, KeySchemaElement, KeyType, Projection, ProjectionType,
    ScalarAttributeType,
};

use crate::KeySchema;
use crate::aws::AwsDynamoDb;
use crate::conformance::{Fixture, Harness, gsi1, lsi1, primary_index};

const DEFAULT_ENDPOINT: &str = "http://localhost:8000";

/// Distinguishes tables created by concurrently running tests within one process.
static TABLE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(crate) struct DynamoDbLocalHarness;

/// A table name that cannot collide with another test in this process, or with a leftover table
/// from a previous run against the same server.
fn unique_table_name() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before the unix epoch")
        .as_nanos();
    let sequence = TABLE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("conformance-{nanos}-{sequence}")
}

fn attribute(name: &str) -> AttributeDefinition {
    AttributeDefinition::builder()
        .attribute_name(name)
        .attribute_type(ScalarAttributeType::S)
        .build()
        .expect("attribute definition is complete")
}

fn key_element(name: &str, key_type: KeyType) -> KeySchemaElement {
    KeySchemaElement::builder()
        .attribute_name(name)
        .key_type(key_type)
        .build()
        .expect("key schema element is complete")
}

/// Creates a table matching `keys`, carrying the secondary indexes the suite expects.
///
/// [`gsi1`] is always present. [`lsi1`] is only created for a composite primary key, since
/// DynamoDB rejects an LSI on a table without a sort key — which is why the suite's LSI test
/// only ever runs against a composite fixture.
async fn create_table(client: &aws_sdk_dynamodb::Client, table: &str, keys: &KeySchema) {
    let gsi1 = gsi1();
    let lsi1 = lsi1();

    // Every attribute used as a key by the table or any of its indexes must be declared, and
    // only those - DynamoDB rejects definitions for attributes that aren't index keys.
    let mut attributes = vec![
        attribute(&keys.pk_identifier),
        attribute(&gsi1.keys.pk_identifier),
    ];
    if let Some(gsi_sk) = &gsi1.keys.sk_identifier {
        attributes.push(attribute(gsi_sk));
    }
    let mut key_schema = vec![key_element(&keys.pk_identifier, KeyType::Hash)];
    if let Some(sk_identifier) = &keys.sk_identifier {
        attributes.push(attribute(sk_identifier));
        attributes.push(attribute(&lsi1.sk_identifier));
        key_schema.push(key_element(sk_identifier, KeyType::Range));
    }

    // The suite deserializes whole payloads out of index queries, so every index projects all
    // attributes.
    let projection = Projection::builder()
        .projection_type(ProjectionType::All)
        .build();

    let mut gsi_definition = aws_sdk_dynamodb::types::GlobalSecondaryIndex::builder()
        .index_name(&gsi1.name)
        .key_schema(key_element(&gsi1.keys.pk_identifier, KeyType::Hash))
        .projection(projection.clone());
    if let Some(gsi_sk) = &gsi1.keys.sk_identifier {
        gsi_definition = gsi_definition.key_schema(key_element(gsi_sk, KeyType::Range));
    }

    let mut request = client
        .create_table()
        .table_name(table)
        .billing_mode(BillingMode::PayPerRequest)
        .set_attribute_definitions(Some(attributes))
        .set_key_schema(Some(key_schema))
        .global_secondary_indexes(gsi_definition.build().expect("GSI definition is complete"));

    if keys.sk_identifier.is_some() {
        let lsi_definition = aws_sdk_dynamodb::types::LocalSecondaryIndex::builder()
            .index_name(&lsi1.name)
            .key_schema(key_element(&keys.pk_identifier, KeyType::Hash))
            .key_schema(key_element(&lsi1.sk_identifier, KeyType::Range))
            .projection(projection)
            .build()
            .expect("LSI definition is complete");
        request = request.local_secondary_indexes(lsi_definition);
    }

    request.send().await.unwrap_or_else(|e| {
        panic!(
            "failed to create table {table} on DynamoDB Local. Is it running? \
             Start it with `docker run --rm -d -p 8000:8000 amazon/dynamodb-local`. Error: {}",
            e.into_service_error()
        )
    });
}

impl Harness for DynamoDbLocalHarness {
    type Db = AwsDynamoDb;

    async fn setup(&self, keys: KeySchema) -> Fixture<Self::Db> {
        let endpoint = std::env::var("DYNAMODB_LOCAL_ENDPOINT")
            .unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string());

        let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .endpoint_url(endpoint)
            .region(aws_config::Region::new("local"))
            .credentials_provider(aws_sdk_dynamodb::config::Credentials::new(
                "local",
                "local",
                None,
                None,
                "conformance",
            ))
            .load()
            .await;

        let client = aws_sdk_dynamodb::Client::new(&config);
        let table = unique_table_name();
        let other_table = unique_table_name();
        create_table(&client, &table, &keys).await;
        create_table(&client, &other_table, &keys).await;

        Fixture {
            db: AwsDynamoDb::from_client(client, primary_index(keys)),
            table,
            other_table,
        }
    }
}

crate::conformance::conformance_suite!(DynamoDbLocalHarness);
