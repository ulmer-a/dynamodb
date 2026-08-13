use aws_sdk_dynamodb::types::AttributeValue;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{AwsDynamoDbService, Container, Keys};

/// AWS-backed [`AwsDynamoDbService`] implementation talking to a real DynamoDB table.
#[derive(Debug, Clone)]
pub struct AwsDynamoDb {
    client: aws_sdk_dynamodb::Client,
}

impl AwsDynamoDb {
    /// Creates a new adapter using an existing client.
    pub fn from_client(client: aws_sdk_dynamodb::Client) -> Self {
        Self { client }
    }

    /// Creates a new adapter.
    ///
    /// Takes a shared [`aws_config::SdkConfig`] and builds the DynamoDB client internally, so
    /// callers don't need to depend on `aws-sdk-dynamodb` directly. The target table is
    /// supplied per call to [`AwsDynamoDbService::get_item`].
    pub fn new(config: &aws_config::SdkConfig) -> Self {
        Self {
            client: aws_sdk_dynamodb::Client::new(config),
        }
    }
}

#[async_trait::async_trait]
impl AwsDynamoDbService for AwsDynamoDb {
    async fn get_item<T>(&self, keys: Keys, table: &str) -> Result<Option<(u64, T)>, String>
    where
        T: DeserializeOwned + Send,
    {
        let result = self
            .client
            .get_item()
            .table_name(table)
            .key("pk", AttributeValue::S(keys.pk.clone()))
            .key("sk", AttributeValue::S(keys.sk.clone()))
            .send()
            .await
            .map_err(|e| {
                format!(
                    "DynamoDB GetItem failed for pk={}, sk={}: {e}",
                    keys.pk, keys.sk
                )
            })?;

        result.item.map_or(Ok(None), |hashmap| {
            // Deserialize container
            let container: Container<T> = serde_dynamo::from_item(hashmap).map_err(|e| {
                format!(
                    "Failed to deserialize item for pk={}, sk={}: {e}",
                    keys.pk, keys.sk
                )
            })?;

            Ok(Some((container.data_version, container.payload)))
        })
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
        let container = Container {
            keys: keys.clone(),
            data_version: 1,
            payload,
        };

        let item = serde_dynamo::to_item(&container).map_err(|e| {
            format!(
                "Failed to serialize item for pk={}, sk={}: {e}",
                keys.pk, keys.sk
            )
        })?;

        self.client
            .put_item()
            .table_name(table)
            .set_item(Some(item))
            .send()
            .await
            .map_err(|e| {
                format!(
                    "DynamoDB PutItem failed for pk={}, sk={}: {e}",
                    keys.pk, keys.sk
                )
            })?;

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
            return Err(
                "put_item() does not accept version=0. Use put_item_unconditional()".to_string(),
            );
        }

        let new_version = expected_version + 1;

        let container = Container {
            keys: keys.clone(),
            data_version: new_version,
            payload,
        };

        let item = serde_dynamo::to_item(&container).map_err(|e| {
            format!(
                "Failed to serialize item for pk={}, sk={}: {e}",
                keys.pk, keys.sk
            )
        })?;

        let request = self
            .client
            .put_item()
            .table_name(table)
            .set_item(Some(item))
            .condition_expression("data_version = :expected")
            .expression_attribute_values(
                ":expected",
                AttributeValue::N(expected_version.to_string()),
            );

        request.send().await.map_err(|e| {
            let err = e.into_service_error();
            if err.is_conditional_check_failed_exception() {
                format!(
                    "Optimistic concurrency conflict for pk={}, sk={}: expected version {expected_version:?}",
                    keys.pk, keys.sk
                )
            } else {
                format!(
                    "DynamoDB PutItem failed for pk={}, sk={}: {err}",
                    keys.pk, keys.sk
                )
            }
        })?;

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
        let mut items = Vec::new();
        let mut last_evaluated_key = None;

        // DynamoDB pages large result sets, so keep querying until the table stops handing back
        // a continuation key.
        loop {
            let result = self
                .client
                .query()
                .table_name(table)
                .key_condition_expression("pk = :pk AND begins_with(sk, :sk_prefix)")
                .expression_attribute_values(":pk", AttributeValue::S(pk.clone()))
                .expression_attribute_values(":sk_prefix", AttributeValue::S(sk_prefix.clone()))
                .set_exclusive_start_key(last_evaluated_key)
                .send()
                .await
                .map_err(|e| {
                    format!("DynamoDB Query failed for pk={pk}, sk_prefix={sk_prefix}: {e}")
                })?;

            for hashmap in result.items.unwrap_or_default() {
                let container: Container<T> = serde_dynamo::from_item(hashmap).map_err(|e| {
                    format!("Failed to deserialize item for pk={pk}, sk_prefix={sk_prefix}: {e}")
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

    async fn delete_item(&self, keys: Keys, table: &str) -> Result<(), String> {
        self.client
            .delete_item()
            .table_name(table)
            .key("pk", AttributeValue::S(keys.pk.clone()))
            .key("sk", AttributeValue::S(keys.sk.clone()))
            .send()
            .await
            .map_err(|e| {
                format!(
                    "DynamoDB DeleteItem failed for pk={}, sk={}: {e}",
                    keys.pk, keys.sk
                )
            })?;

        Ok(())
    }
}
