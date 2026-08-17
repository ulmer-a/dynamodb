# Changelog

## 0.3.0

A breaking release, centred on making a read-modify-write over a key that may not exist safe.

### The problem

`put_item` could not create an item — its condition compares an attribute of an existing one — so
bootstrapping had to go through `put_item_unconditional`, which had no condition at all. Two
callers that both observed a key as absent could therefore both "create" it, and the slower one's
write silently destroyed everything the faster one had committed. No call returned an error. The
window is narrow, but it is the bootstrap path, so it lands on an empty table.

### Added

- `create_item`: writes only if the key is absent (`attribute_not_exists`), returning
  `Error::AlreadyExists` otherwise. This is the safe way to bootstrap, and closes the lost-update
  race above.
- `add_to_counter`: atomically adds to a numeric attribute in a single `UpdateItem`, returning the
  new value. Concurrent callers each get a distinct number and none can lose an increment, so
  allocating identifiers needs no retry loop. Bumps `data_version` in the same request.
- `INITIAL_DATA_VERSION`, the version a newly created item starts at.

### Changed

- **Errors are typed.** Every operation returns `Result<_, Error>` instead of `Result<_, String>`.
  `Error` is `#[non_exhaustive]` with `AlreadyExists`, `Conflict`, `InvalidRequest`, `Unsupported`,
  `Serialization` and `Service`. Callers previously had to substring-match
  `"Optimistic concurrency conflict"` to tell a lost race from a service failure; such code now
  fails to compile rather than silently stopping to match.
- **`put_item` accepts `expected_version` of `0`**, meaning "the item exists but carries no
  `data_version`" — a record written before versioning existed. The write upgrades it onto the
  versioning scheme. Previously `0` was rejected outright, which left those records writable only
  through the unconditional path. `0` does *not* mean "create if absent"; that is `create_item`.
- **`data_version` is now monotonic per key.** `put_item_unconditional` advances the version past
  whatever was stored instead of resetting it to `1`. Resetting made a spent compare-and-swap
  token valid again against a later generation of the item — an ABA bug that let a stale CAS
  succeed. It now returns the new version, and on the AWS backend costs a read.
- **`MockDynamoDb::new` takes a `PrimaryIndex`.** The mock now validates `sk` against a key schema
  the way the AWS backend always has; previously it accepted keys the real backend rejected.
- `Container` and `MockDbKey` are no longer public. Neither could be constructed or read from
  outside the crate, and neither appeared in any public signature.

### Deprecated

- `put_item_unconditional`. `create_item` and `put_item` between them cover every case, and both
  report a lost race, which it cannot.

### Testing

- Added a backend-agnostic conformance suite. Every test is generic over the backend, so the same
  assertions run against `MockDynamoDb` and — behind the `dynamodb-local-tests` feature — against
  a real DynamoDB engine. The mock is only useful insofar as it behaves like the backend it stands
  in for, and it had already drifted: it accepted keys the AWS backend rejects.

## 0.2.0

- Support arbitrary GSIs/LSIs with configurable key attribute names.
