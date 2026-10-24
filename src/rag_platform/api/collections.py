"""Collections (plan section I): named groups of documents - shared or private.

    GET    /collections                  the collections the caller may search (open)
    GET    /collections/private-summary  totals over all private collections (admin)
    POST   /collections                  create one: shared (admin) or private (signed-in user)
    PUT    /collections/{id}             rename / describe it (who may manage it)
    DELETE /collections/{id}             delete it with its documents and chats (who may manage it)

Why: unrelated domains (an AI handbook, HR policies, a loan file) shouldn't be searched together.
A chat picks one collection and only ever searches it: each collection has its own FAISS index
file and its own BM25 keyword index (Settings.collection_index_path, SearchCache), so isolation
holds by construction, not by filtering results afterwards.

Shared vs private (phase 2, rules in deps.can_see / can_manage):
- shared collections are managed by admins and searchable by everyone, guests included;
- any signed-in user can create PRIVATE collections, upload into them and ask questions; nobody
  else sees them - other users get 404 (as if they didn't exist), admins only see totals
  (private-summary), never names, documents or answers. Per-user limits (USER_MAX_*) protect the
  shared embedding quota.
Collection 1 is the demo collection: evaluation, the optimizer and the CI gate work on it, so
it can be renamed but not deleted.
"""

import shutil
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, status

from rag_platform.api import service
from rag_platform.api.deps import (
    Caller,
    Services,
    can_manage,
    can_see,
    current_caller,
    get_services,
    manageable_collection,
    optional_caller,
    private_owner,
    require_admin,
    write_lock,
)
from rag_platform.api.schemas import Collection, CollectionIn, PrivateSummary
from rag_platform.config import DEFAULT_COLLECTION_ID

router = APIRouter(tags=["collections"])


def _out(svc: Services, store, row: dict, caller: Caller | None) -> Collection:
    threshold, source = service.threshold_for(svc, store, row)
    return Collection(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        documents=row["documents"],
        passages=row["passages"],
        is_default=row["id"] == DEFAULT_COLLECTION_ID,
        is_private=row["owner"] != "",
        can_manage=can_manage(caller, row),
        threshold=threshold,
        threshold_source=source,
        threshold_override=row["threshold"],
    )


def _one(svc: Services, collection_id: int, caller: Caller) -> Collection:
    store = svc.open_store()
    try:
        row = next((c for c in store.list_collections() if c["id"] == collection_id), None)
        if row is None or not can_see(caller, row):
            raise HTTPException(404, f"No collection with id {collection_id}")
        return _out(svc, store, row, caller)
    finally:
        store.close()


@router.get("/collections", response_model=list[Collection])
def list_collections(
    caller: Caller | None = Depends(optional_caller), svc: Services = Depends(get_services)
) -> list[Collection]:
    """Open, like /assistant (the chat's picker needs it for guests too): shared collections for
    everyone, plus the signed-in caller's own private ones."""
    store = svc.open_store()
    try:
        rows = [c for c in store.list_collections() if can_see(caller, c)]
        return [_out(svc, store, c, caller) for c in rows]
    finally:
        store.close()


@router.get(
    "/collections/private-summary",
    response_model=PrivateSummary,
    dependencies=[Depends(require_admin)],
)
def private_summary(svc: Services = Depends(get_services)) -> PrivateSummary:
    """All an admin may know about users' private collections: how many, never what."""
    store = svc.open_store()
    try:
        return PrivateSummary(**store.private_totals())
    finally:
        store.close()


@router.post("/collections", response_model=Collection, status_code=status.HTTP_201_CREATED)
def create_collection(
    body: CollectionIn,
    caller: Caller = Depends(current_caller),
    svc: Services = Depends(get_services),
) -> Collection:
    """Admins create shared collections (or a private one of their own with private=true);
    other signed-in users always create private ones; guests can't create any."""
    owner = private_owner(caller)
    private = body.private or not caller.is_admin
    if private and owner is None:
        raise HTTPException(403, "Sign in with Google to create your own collections")
    s = svc.settings
    store = svc.open_store()
    try:
        if private and store.owner_usage(owner)["collections"] >= s.user_max_collections:
            raise HTTPException(
                403, f"You can have at most {s.user_max_collections} private collections"
            )
        name, description = body.name.strip(), body.description.strip()
        collection_id = store.create_collection(name, description, owner if private else "")
        if body.threshold is not None:
            store.update_collection(collection_id, name, description, body.threshold)
    except sqlite3.IntegrityError:  # UNIQUE(owner, name)
        raise HTTPException(409, f'A collection named "{body.name}" already exists') from None
    finally:
        store.close()
    return _one(svc, collection_id, caller)


@router.put("/collections/{collection_id}", response_model=Collection)
def update_collection(
    collection_id: int,
    body: CollectionIn,
    caller: Caller = Depends(current_caller),
    svc: Services = Depends(get_services),
) -> Collection:
    if collection_id == DEFAULT_COLLECTION_ID and body.threshold is not None:
        raise HTTPException(
            422, "The demo collection's threshold changes only through the optimizer (Settings)"
        )
    store = svc.open_store()
    try:
        manageable_collection(store, caller, collection_id)
        store.update_collection(
            collection_id, body.name.strip(), body.description.strip(), body.threshold
        )
    except sqlite3.IntegrityError:
        raise HTTPException(409, f'A collection named "{body.name}" already exists') from None
    finally:
        store.close()
    return _one(svc, collection_id, caller)


@router.delete("/collections/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(
    collection_id: int,
    caller: Caller = Depends(current_caller),
    svc: Services = Depends(get_services),
) -> None:
    """Delete the collection, its documents (passages), its chats, its search index and the
    files uploaded into it. The demo collection can't be deleted (evaluation depends on it)."""
    if collection_id == DEFAULT_COLLECTION_ID:
        raise HTTPException(409, "The demo collection can't be deleted (evaluation uses it)")
    with write_lock:  # no upload into it while it disappears
        store = svc.open_store()
        try:
            manageable_collection(store, caller, collection_id)
            remove_collection(svc, store, collection_id)
        finally:
            store.close()


def remove_collection(svc: Services, store, collection_id: int) -> None:
    """Delete a collection everywhere: database rows, search index, uploaded files, cache.
    Also used by "Delete my account" for every private collection of the user."""
    s = svc.settings
    store.delete_collection(collection_id)
    s.collection_index_path(collection_id).unlink(missing_ok=True)
    shutil.rmtree(s.upload_dir / str(collection_id), ignore_errors=True)
    svc.search.forget(collection_id)
