use std::collections::HashMap;

use aws_sdk_dynamodb::types::{AttributeValue, ReturnValue};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{
    AnyIndex, AwsDynamoDbService, Container, DATA_VERSION_ATTRIBUTE, Error, INITIAL_DATA_VERSION,
    PrimaryIndex, SortKeyCondition,
};

/// How many times [`AwsDynamoDbService::put_item_unconditional`] will re-read and retry before
/// giving up.
///
/// It has to read the current version to advance past it, which opens a window another writer
/// can land in. Each retry closes on a fresher read, so exhausting this budget means sustained
/// contention rather than an unlucky interleaving.
const UNCONDITIONAL_WRITE_ATTEMPTS: usize = 8;

/// The guard placed on a conditional PutItem.
#[derive(Debug, Clone, Copy)]
enum WriteCondition {
    /// No item may exist under the key.
    MustNotExist,
    /// An item must exist and its stored `data_version` must be exactly this.
    VersionIs(u64),
}

/// AWS-backed [`AwsDynamoDbService`] implementation talking to a real DynamoDB table.
#[derive(Debug, Clone)]
pub struct AwsDynamoDb {
    client: aws_sdk_dynamodb::Client,
    primary_index: PrimaryIndex,
}

impl AwsDynamoDb {
    /// Creates a new adapter using an existing client.
    ///
    /// `primary_index` describes the attribute names of the base table's primary key; it's
    /// used for every [`AwsDynamoDbService::get_item`]/`put_item`/`delete_item` call.
    pub fn from_client(client: aws_sdk_dynamodb::Client, primary_index: PrimaryIndex) -> Self {
        Self {
            client,
            primary_index,
        }
    }

    /// Creates a new adapter.
    ///
    /// Takes a shared [`aws_config::SdkConfig`] and builds the DynamoDB client internally, so
    /// callers don't need to depend on `aws-sdk-dynamodb` directly. The target table is
    /// supplied per call to [`AwsDynamoDbService::get_item`].
    pub fn new(config: &aws_config::SdkConfig, primary_index: PrimaryIndex) -> Self {
        Self {
            client: aws_sdk_dynamodb::Client::new(config),
            primary_index,
        }
    }

    /// Resolves `sk` against the configured [`PrimaryIndex`], returning the sort key's
    /// attribute name paired with the value when the table has one. `Err` if `sk` doesn't
    /// match whether the table actually has a sort key.
    fn resolve_sk(&self, sk: &Option<String>) -> Result<Option<(&str, String)>, Error> {
        self.primary_index.resolve_sk(sk)
    }

    /// Builds the DynamoDB item for `payload` at `data_version`, with the primary key attributes
    /// added under the names the configured [`PrimaryIndex`] specifies.
    fn build_item<T>(
        &self,
        pk: &str,
        sk_attr: &Option<(&str, String)>,
        data_version: u64,
        payload: &T,
    ) -> Result<HashMap<String, AttributeValue>, Error>
    where
        T: Serialize,
    {
        let container = Container {
            data_version,
            payload,
        };

        let mut item: HashMap<String, AttributeValue> =
            serde_dynamo::to_item(&container).map_err(|e| {
                Error::Serialization(format!("Failed to serialize item for pk={pk}: {e}"))
            })?;
        item.insert(
            self.primary_index.keys.pk_identifier.clone(),
            AttributeValue::S(pk.to_string()),
        );
        if let Some((sk_identifier, sk_value)) = sk_attr {
            item.insert(
                sk_identifier.to_string(),
                AttributeValue::S(sk_value.clone()),
            );
        }
        Ok(item)
    }

    /// PutItem guarded by `condition`, translating a failed conditional check into whichever
    /// error that particular guard means.
    async fn put_guarded(
        &self,
        pk: &str,
        sk: &Option<String>,
        table: &str,
        item: HashMap<String, AttributeValue>,
        condition: WriteCondition,
    ) -> Result<(), Error> {
        let mut request = self
            .client
            .put_item()
            .table_name(table)
            .set_item(Some(item));

        // Only the names and values a given expression actually references may be declared —
        // DynamoDB rejects the request outright if any go unused. The partition key goes through
        // an expression attribute name because its attribute name is caller-configurable and
        // could collide with a DynamoDB reserved word.
        request = match condition {
            // Evaluated against the item currently stored under the key, so when there is none,
            // no attribute exists and the condition passes.
            WriteCondition::MustNotExist => request
                .condition_expression("attribute_not_exists(#pk)")
                .expression_attribute_names("#pk", self.primary_index.keys.pk_identifier.as_str()),

            // A record written before versioning existed carries no data_version attribute at
            // all, and DynamoDB evaluates a comparison against an absent attribute as false — so
            // version 0 has to be matched by absence rather than by equality.
            WriteCondition::VersionIs(0) => request
                .condition_expression("attribute_exists(#pk) AND attribute_not_exists(#version)")
                .expression_attribute_names("#pk", self.primary_index.keys.pk_identifier.as_str())
                .expression_attribute_names("#version", DATA_VERSION_ATTRIBUTE),

            WriteCondition::VersionIs(expected) => request
                .condition_expression("#version = :expected")
                .expression_attribute_names("#version", DATA_VERSION_ATTRIBUTE)
                .expression_attribute_values(":expected", AttributeValue::N(expected.to_string())),
        };

        request.send().await.map_err(|e| {
            let err = e.into_service_error();
            if !err.is_conditional_check_failed_exception() {
                return Error::Service(format!(
                    "DynamoDB PutItem failed for pk={pk}, sk={sk:?}: {err}"
                ));
            }
            // A failed conditional check is the guard doing its job, not the service failing.
            match condition {
                WriteCondition::MustNotExist => Error::AlreadyExists {
                    pk: pk.to_string(),
                    sk: sk.clone(),
                },
                // DynamoDB doesn't report the version it actually found, so Error::Conflict
                // can't carry it.
                WriteCondition::VersionIs(expected_version) => Error::Conflict {
                    pk: pk.to_string(),
                    sk: sk.clone(),
                    expected_version,
                },
            }
        })?;

        Ok(())
    }

    /// Reads just the stored `data_version` for `(pk, sk)`.
    ///
    /// `None` if no item exists; `Some(0)` if one exists but predates versioning. Projects the
    /// partition key alongside the version so an unversioned item is still distinguishable from
    /// an absent one, and reads consistently so the compare-and-swap it feeds isn't racing a
    /// replica lag it could have avoided.
    async fn current_data_version(
        &self,
        pk: &str,
        sk_attr: &Option<(&str, String)>,
        table: &str,
    ) -> Result<Option<u64>, Error> {
        let mut request = self
            .client
            .get_item()
            .table_name(table)
            .consistent_read(true)
            .projection_expression("#pk, #version")
            .expression_attribute_names("#pk", self.primary_index.keys.pk_identifier.as_str())
            .expression_attribute_names("#version", DATA_VERSION_ATTRIBUTE)
            .key(
                self.primary_index.keys.pk_identifier.as_str(),
                AttributeValue::S(pk.to_string()),
            );
        if let Some((sk_identifier, sk_value)) = sk_attr {
            request = request.key(*sk_identifier, AttributeValue::S(sk_value.clone()));
        }

        let result = request
            .send()
            .await
            .map_err(|e| Error::Service(format!("DynamoDB GetItem failed for pk={pk}: {e}")))?;

        let Some(item) = result.item else {
            return Ok(None);
        };

        let version = match item.get(DATA_VERSION_ATTRIBUTE) {
            Some(AttributeValue::N(n)) => n.parse::<u64>().map_err(|e| {
                Error::Serialization(format!(
                    "Stored {DATA_VERSION_ATTRIBUTE} for pk={pk} is not a u64: {e}"
                ))
            })?,
            // Present but unversioned: a legacy record, which reads as version 0.
            _ => 0,
        };
        Ok(Some(version))
    }
}

#[async_trait::async_trait]
impl AwsDynamoDbService for AwsDynamoDb {
    async fn get_item<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
    ) -> Result<Option<(u64, T)>, Error>
    where
        T: DeserializeOwned + Send,
    {
        let sk_attr = self.resolve_sk(&sk)?;

        let mut request = self.client.get_item().table_name(table).key(
            self.primary_index.keys.pk_identifier.as_str(),
            AttributeValue::S(pk.clone()),
        );
        if let Some((sk_identifier, sk_value)) = sk_attr {
            request = request.key(sk_identifier, AttributeValue::S(sk_value));
        }

        let result = request.send().await.map_err(|e| {
            Error::Service(format!(
                "DynamoDB GetItem failed for pk={pk}, sk={sk:?}: {e}"
            ))
        })?;

        result.item.map_or(Ok(None), |hashmap| {
            // Deserialize container
            let container: Container<T> = serde_dynamo::from_item(hashmap).map_err(|e| {
                Error::Serialization(format!(
                    "Failed to deserialize item for pk={pk}, sk={sk:?}: {e}"
                ))
            })?;

            Ok(Some((container.data_version, container.payload)))
        })
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
        let sk_attr = self.resolve_sk(&sk)?;
        let item = self.build_item(&pk, &sk_attr, INITIAL_DATA_VERSION, payload)?;

        self.put_guarded(&pk, &sk, table, item, WriteCondition::MustNotExist)
            .await?;

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
        let sk_attr = self.resolve_sk(&sk)?;

        // PutItem replaces the whole item and condition expressions can't compute a value, so
        // there is no single request that both overwrites the payload and advances the version.
        // (UpdateItem could bump the version atomically, but it merges rather than replaces,
        // which would silently keep attributes from an older payload shape.) So: read the
        // current version, write conditioned on it still being current, and retry if it moved.
        let mut last_seen = 0;
        for _ in 0..UNCONDITIONAL_WRITE_ATTEMPTS {
            let current = self.current_data_version(&pk, &sk_attr, table).await?;
            last_seen = current.unwrap_or(0);

            let (new_version, condition) = match current {
                None => (INITIAL_DATA_VERSION, WriteCondition::MustNotExist),
                Some(version) => (version + 1, WriteCondition::VersionIs(version)),
            };

            let item = self.build_item(&pk, &sk_attr, new_version, payload)?;
            match self.put_guarded(&pk, &sk, table, item, condition).await {
                Ok(()) => return Ok(new_version),
                // Someone wrote between our read and our write. Both guards report that the
                // same way: re-read and try again.
                Err(Error::AlreadyExists { .. } | Error::Conflict { .. }) => continue,
                Err(other) => return Err(other),
            }
        }

        Err(Error::Conflict {
            pk,
            sk,
            expected_version: last_seen,
        })
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
        let sk_attr = self.resolve_sk(&sk)?;
        // Also correct for expected_version 0: an unversioned record lands at
        // INITIAL_DATA_VERSION. WriteCondition renders that guard as an absence check rather
        // than an equality one.
        let new_version = expected_version + 1;
        let item = self.build_item(&pk, &sk_attr, new_version, payload)?;

        self.put_guarded(
            &pk,
            &sk,
            table,
            item,
            WriteCondition::VersionIs(expected_version),
        )
        .await?;

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
        let sk_attr = self.resolve_sk(&sk)?;

        let mut request = self
            .client
            .update_item()
            .table_name(table)
            .key(
                self.primary_index.keys.pk_identifier.as_str(),
                AttributeValue::S(pk.clone()),
            )
            // ADD treats a missing attribute as 0, so this both creates the item when the key is
            // absent and upgrades an unversioned one: data_version lands on INITIAL_DATA_VERSION
            // either way. Bumping it in the same request keeps it monotonic without a read.
            .update_expression("ADD #counter :delta, #version :one")
            .expression_attribute_names("#counter", counter_attribute)
            .expression_attribute_names("#version", DATA_VERSION_ATTRIBUTE)
            .expression_attribute_values(":delta", AttributeValue::N(delta.to_string()))
            .expression_attribute_values(
                ":one",
                AttributeValue::N(INITIAL_DATA_VERSION.to_string()),
            )
            .return_values(ReturnValue::UpdatedNew);
        if let Some((sk_identifier, sk_value)) = sk_attr {
            request = request.key(sk_identifier, AttributeValue::S(sk_value));
        }

        let result = request.send().await.map_err(|e| {
            Error::Service(format!(
                "DynamoDB UpdateItem failed for pk={pk}, sk={sk:?}: {}",
                e.into_service_error()
            ))
        })?;

        // UPDATED_NEW returns exactly the attributes this expression touched.
        let attributes = result.attributes.unwrap_or_default();
        match attributes.get(counter_attribute) {
            Some(AttributeValue::N(n)) => n.parse::<i64>().map_err(|e| {
                Error::Serialization(format!(
                    "Counter {counter_attribute:?} for pk={pk} is not an i64: {e}"
                ))
            }),
            other => Err(Error::Service(format!(
                "DynamoDB UpdateItem did not return a numeric {counter_attribute:?} for pk={pk}: \
                 {other:?}"
            ))),
        }
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
        let pk_identifier = index.pk_identifier(&self.primary_index);
        let sk_identifier = index.sk_identifier();

        if sk_condition.is_some() && sk_identifier.is_none() {
            return Err(Error::sk_condition_without_sort_key());
        }

        let mut key_condition = "#pk = :pk".to_string();
        let mut names = vec![("#pk".to_string(), pk_identifier.to_string())];
        let mut values = vec![(":pk".to_string(), AttributeValue::S(pk.clone()))];

        if let (Some(sk_identifier), Some(condition)) = (sk_identifier, &sk_condition) {
            names.push(("#sk".to_string(), sk_identifier.to_string()));
            match condition {
                SortKeyCondition::Prefix(prefix) => {
                    key_condition.push_str(" AND begins_with(#sk, :sk_value)");
                    values.push((":sk_value".to_string(), AttributeValue::S(prefix.clone())));
                }
                SortKeyCondition::Between(low, high) => {
                    key_condition.push_str(" AND #sk BETWEEN :sk_low AND :sk_high");
                    values.push((":sk_low".to_string(), AttributeValue::S(low.clone())));
                    values.push((":sk_high".to_string(), AttributeValue::S(high.clone())));
                }
                other => {
                    return Err(Error::Unsupported(format!(
                        "{other:?} is not yet implemented by AwsDynamoDb"
                    )));
                }
            }
        }

        let mut items = Vec::new();
        let mut last_evaluated_key = None;

        // DynamoDB pages large result sets, so keep querying until the table stops handing back
        // a continuation key.
        loop {
            let mut request = self
                .client
                .query()
                .table_name(table)
                .set_index_name(index.index_name().map(String::from))
                .key_condition_expression(key_condition.clone())
                .set_exclusive_start_key(last_evaluated_key);

            for (name, value) in &names {
                request = request.expression_attribute_names(name, value);
            }
            for (name, value) in &values {
                request = request.expression_attribute_values(name, value.clone());
            }

            let result = request
                .send()
                .await
                .map_err(|e| Error::Service(format!("DynamoDB Query failed for pk={pk}: {e}")))?;

            for hashmap in result.items.unwrap_or_default() {
                let container: Container<T> = serde_dynamo::from_item(hashmap).map_err(|e| {
                    Error::Serialization(format!("Failed to deserialize item for pk={pk}: {e}"))
                })?;
                items.push((container.data_version, container.payload));
            }

            last_evaluated_key = result.last_evaluated_key;
            if last_evaluated_key.is_none() {
                break;
            }
        }

        Ok(items)
    }

    async fn delete_item(&self, pk: String, sk: Option<String>, table: &str) -> Result<(), Error> {
        let sk_attr = self.resolve_sk(&sk)?;

        let mut request = self.client.delete_item().table_name(table).key(
            self.primary_index.keys.pk_identifier.as_str(),
            AttributeValue::S(pk.clone()),
        );
        if let Some((sk_identifier, sk_value)) = sk_attr {
            request = request.key(sk_identifier, AttributeValue::S(sk_value));
        }

        request.send().await.map_err(|e| {
            Error::Service(format!(
                "DynamoDB DeleteItem failed for pk={pk}, sk={sk:?}: {e}"
            ))
        })?;

        Ok(())
    }
}
