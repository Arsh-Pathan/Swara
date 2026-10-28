"""Unit and integration tests for speaker enrollment and identification."""

import numpy as np
import pytest

from swara.core.audio import load_audio
from swara.core.types import AudioChunk, EnrolledSpeaker
from swara.embedding.wespeaker import WeSpeakerEncoder
from swara.identification.engine import SpeakerIdentificationEngine
from swara.identification.store import SpeakerStore


def test_enrollment_and_identification(temp_dir):
    store = SpeakerStore(storage_dir=temp_dir / "speakers")
    encoder = WeSpeakerEncoder()
    engine = SpeakerIdentificationEngine(encoder=encoder, store=store, default_threshold=0.65)

    # Real voice sample Alice
    audio_alice = load_audio("samples/speechbrain_example1.wav")
    sr = 16000
    clip_alice_1 = audio_alice.samples[: int(1.8 * sr)]
    clip_alice_2 = audio_alice.samples[int(1.8 * sr) :]

    # Enroll Alice
    profile = engine.enroll(speaker_id="alice", audio=clip_alice_1, name="Alice User")
    assert profile.speaker_id == "alice"
    assert profile.name == "Alice User"

    # Identify Alice using second clip
    match_res = engine.identify(audio=clip_alice_2)
    assert match_res.is_match is True
    assert match_res.speaker_id == "alice"
    assert match_res.similarity > 0.65

    # Identify an un-enrolled voice (Bob from en_speech) -> Expect UNKNOWN
    audio_bob = load_audio("samples/en_speech.wav")
    clip_bob = audio_bob.samples[int(0.5 * sr) : int(3.5 * sr)]
    match_bob = engine.identify(audio=clip_bob)
    assert match_bob.is_match is False
    assert match_bob.speaker_id == "UNKNOWN"


def test_store_persistence_survives_restart(temp_dir):
    store_dir = temp_dir / "speakers_persist"
    encoder = WeSpeakerEncoder()

    # Session 1: Enroll speaker
    store1 = SpeakerStore(storage_dir=store_dir)
    engine1 = SpeakerIdentificationEngine(encoder=encoder, store=store1)
    audio = load_audio("samples/speechbrain_example1.wav")
    engine1.enroll(speaker_id="charlie", audio=audio, name="Charlie")

    # Session 2: Fresh store instance loading from same directory
    store2 = SpeakerStore(storage_dir=store_dir)
    assert len(store2.list_speakers()) == 1
    spk = store2.get_speaker("charlie")
    assert spk is not None
    assert spk.name == "Charlie"

    emb = store2.get_embedding("charlie")
    assert emb is not None
    assert emb.shape == (256,)


def test_speaker_deletion(temp_dir):
    store = SpeakerStore(storage_dir=temp_dir / "speakers_del")
    encoder = WeSpeakerEncoder()
    engine = SpeakerIdentificationEngine(encoder=encoder, store=store)

    audio = load_audio("samples/speechbrain_example1.wav")
    engine.enroll(speaker_id="temp_user", audio=audio)
    assert store.get_speaker("temp_user") is not None

    deleted = store.delete_speaker("temp_user")
    assert deleted is True
    assert store.get_speaker("temp_user") is None
    assert store.get_embedding("temp_user") is None
