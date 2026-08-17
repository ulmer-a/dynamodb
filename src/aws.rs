use std::collections::HashMap;

use aws_sdk_dynamodb::types::AttributeValue;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{AnyIndex, AwsDynamoDbService, Container, PrimaryIndex, SortKeyCondition};

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
    fn resolve_sk(&self, sk: &Option<String>) -> Result<Option<(&str, String)>, String> {
        self.primary_index.resolve_sk(sk)
    }
}

#[async_trait::async_trait]
impl AwsDynamoDbService for AwsDynamoDb {
    async fn get_item<T>(
        &self,
        pk: String,
        sk: Option<String>,
        table: &str,
    ) -> Result<Option<(u64, T)>, String>
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

        let result = request
            .send()
            .await
            .map_err(|e| format!("DynamoDB GetItem failed for pk={pk}, sk={sk:?}: {e}"))?;

        result.item.map_or(Ok(None), |hashmap| {
            // Deserialize container
            let container: Container<T> = serde_dynamo::from_item(hashmap)
                .map_err(|e| format!("Failed to deserialize item for pk={pk}, sk={sk:?}: {e}"))?;

            Ok(Some((container.data_version, container.payload)))
        })
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
        let sk_attr = self.resolve_sk(&sk)?;

        let container = Container {
            data_version: 1,
            payload,
        };

        let mut item: HashMap<String, AttributeValue> = serde_dynamo::to_item(&container)
            .map_err(|e| format!("Failed to serialize item for pk={pk}, sk={sk:?}: {e}"))?;
        item.insert(
            self.primary_index.keys.pk_identifier.clone(),
            AttributeValue::S(pk.clone()),
        );
        if let Some((sk_identifier, sk_value)) = sk_attr {
            item.insert(sk_identifier.to_string(), AttributeValue::S(sk_value));
        }

        self.client
            .put_item()
            .table_name(table)
            .set_item(Some(item))
            .send()
            .await
            .map_err(|e| format!("DynamoDB PutItem failed for pk={pk}, sk={sk:?}: {e}"))?;

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
            return Err(
                "put_item() does not accept version=0. Use put_item_unconditional()".to_string(),
            );
        }

        let sk_attr = self.resolve_sk(&sk)?;
        let new_version = expected_version + 1;

        let container = Container {
            data_version: new_version,
            payload,
        };

        let mut item: HashMap<String, AttributeValue> = serde_dynamo::to_item(&container)
            .map_err(|e| format!("Failed to serialize item for pk={pk}, sk={sk:?}: {e}"))?;
        item.insert(
            self.primary_index.keys.pk_identifier.clone(),
            AttributeValue::S(pk.clone()),
        );
        if let Some((sk_identifier, sk_value)) = sk_attr {
            item.insert(sk_identifier.to_string(), AttributeValue::S(sk_value));
        }

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
                    "Optimistic concurrency conflict for pk={pk}, sk={sk:?}: expected version {expected_version:?}"
                )
            } else {
                format!("DynamoDB PutItem failed for pk={pk}, sk={sk:?}: {err}")
            }
        })?;

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
        let pk_identifier = index.pk_identifier(&self.primary_index);
        let sk_identifier = index.sk_identifier();

        if sk_condition.is_some() && sk_identifier.is_none() {
            return Err(
                "a sort key condition was provided but the index has no sort key".to_string(),
            );
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
                    return Err(format!("{other:?} is not yet implemented by AwsDynamoDb"));
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
                .map_err(|e| format!("DynamoDB Query failed for pk={pk}: {e}"))?;

            for hashmap in result.items.unwrap_or_default() {
                let container: Container<T> = serde_dynamo::from_item(hashmap)
                    .map_err(|e| format!("Failed to deserialize item for pk={pk}: {e}"))?;
                items.push((container.data_version, container.payload));
            }

            last_evaluated_key = result.last_evaluated_key;
            if last_evaluated_key.is_none() {
                break;
            }
        }

        Ok(items)
    }

    async fn delete_item(&self, pk: String, sk: Option<String>, table: &str) -> Result<(), String> {
        let sk_attr = self.resolve_sk(&sk)?;

        let mut request = self.client.delete_item().table_name(table).key(
            self.primary_index.keys.pk_identifier.as_str(),
            AttributeValue::S(pk.clone()),
        );
        if let Some((sk_identifier, sk_value)) = sk_attr {
            request = request.key(sk_identifier, AttributeValue::S(sk_value));
        }

        request
            .send()
            .await
            .map_err(|e| format!("DynamoDB DeleteItem failed for pk={pk}, sk={sk:?}: {e}"))?;

        Ok(())
    }
}
