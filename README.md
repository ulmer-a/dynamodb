# High Level DynamoDB abstraction

A high-level DynamoDB abstraction with optimistic concurrency control built in.

Every item is stored alongside a `data_version`, and the write operations are defined by what
they expect that version to be. `AwsDynamoDbService` is the whole surface: `AwsDynamoDb` talks to
a real table, `MockDynamoDb` is an in-memory stand-in for tests, and a shared conformance suite
holds the two to the same behaviour.

## Choosing a write operation

| Operation | Guard | Use when |
|---|---|---|
| `create_item` | item must not exist | bootstrapping a key |
| `put_item` | stored version must equal `expected_version` | updating a key you have read |
| `add_to_counter` | none; the add is atomic server-side | allocating numbers |
| `put_item_unconditional` | none | deprecated — prefer the above |

## The read-modify-write loop

`get_item` returns the stored version, which feeds straight back into `put_item`. When the key
may not exist yet, `create_item` covers the other branch — the two together are what make the
loop safe, because both report a lost race instead of silently overwriting the winner:

```rust
loop {
    let result = match db.get_item(pk.clone(), sk.clone(), table).await? {
        Some((version, value)) => db
            .put_item(pk.clone(), sk.clone(), table, version, &next(value))
            .await
            .map(|_| ()),
        None => db
            .create_item(pk.clone(), sk.clone(), table, &initial())
            .await
            .map(|_| ()),
    };
    match result {
        Ok(()) => break,
        // Someone else got there first, in either branch. Re-read and try again.
        Err(Error::Conflict { .. } | Error::AlreadyExists { .. }) => continue,
        Err(other) => return Err(other),
    }
}
```

A record written before this crate's versioning existed reads back as version `0`; feeding that
`0` back into `put_item` upgrades it onto the scheme, so the loop needs no special case for it.

To allocate identifiers, `add_to_counter` does it in one round trip with no retry loop, because
the increment happens server-side:

```rust
let catalogue_number = db
    .add_to_counter(pk, sk, table, "next", 1)
    .await?;
```

## Migrating from 0.2

0.3.0 is a breaking release. See [CHANGELOG.md](CHANGELOG.md) for the full list; the changes that
touch every caller are:

- **Errors are typed.** Every operation returns `Result<_, pro_dynamo::Error>` instead of
  `Result<_, String>`. Code that matched on error text — `err.contains("Optimistic concurrency
  conflict")` — now matches `Error::Conflict { .. }` instead, and stops compiling rather than
  silently failing to match.
- **`MockDynamoDb::new` takes a `PrimaryIndex`.** The mock now validates keys against a schema
  the way the AWS backend always has.
- **`put_item_unconditional` is deprecated** and returns the new version. Replace it with
  `create_item` (bootstrap) or `put_item` (update); both report a lost race, which it cannot.
- **`put_item` accepts `expected_version` of `0`**, meaning "exists but unversioned". It used to
  reject `0` outright, which left records predating versioning unwritable.

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

## Licence

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE.txt) or
[MIT licence](LICENSE-MIT.txt) at your option.
