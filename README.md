# High Level DynamoDB abstraction

This repo contains high level dynamodb abstractions.

## Testing

`cargo test` runs the backend-agnostic conformance suite in `src/conformance/` against
`MockDynamoDb`, plus a handful of mock-only fixture tests.

The same suite runs against a real DynamoDB engine behind the `dynamodb-local-tests` feature.
Every assertion the mock is checked against is checked against the engine it stands in for, which
is what keeps the two from drifting apart:

```sh
docker run --rm -d -p 8000:8000 amazon/dynamodb-local
cargo test --features dynamodb-local-tests
```

Set `DYNAMODB_LOCAL_ENDPOINT` to point at a server on another address.

New behaviour shared by both backends belongs in `src/conformance/mod.rs` — add the generic
`async fn` and list its name in the `conformance_suite!` macro at the bottom of that file.