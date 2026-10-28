"""Swara Quickstart Demo Script.

Demonstrates Voice Activity Detection, Speaker Enrollment, Speaker Identification,
and Multi-Speaker Diarization with Identity Enhancement.
"""

from swara import Swara

def main():
    print("=" * 60)
    print("Swara - Speaker Intelligence AI")
    print("CSKY - 'Every Voice Has an Identity.'")
    print("=" * 60)

    # 1. Initialize Swara Engine
    swara = Swara()
    # Reset store for clean demonstration
    for spk in swara.list_speakers():
        swara.delete_speaker(spk.speaker_id)

    print("\n[+] Initialized Swara Engine:")
    print(f"    Backbone Model: {swara.encoder.model_version}")
    print(f"    Embedding Dim:  {swara.encoder.embedding_dim}")

    # 2. Enroll Voice Sample (Alice)
    print("\n[+] Enrolling Voice Sample: 'alice'")
    profile = swara.enroll(
        speaker_id="alice",
        audio_path="samples/speechbrain_example1.wav",
        name="Alice"
    )
    print(f"    Enrolled: {profile.name} (ID: {profile.speaker_id}, Samples: {profile.sample_count})")

    # 3. Identify Voice Sample
    print("\n[+] Identifying Probe Sample: 'samples/alice_test.wav'")
    res = swara.identify(audio_path="samples/alice_test.wav")
    print(f"    Match Result:      {res.speaker_id}")
    print(f"    Verified Match:    {res.is_match}")
    print(f"    Cosine Similarity: {res.similarity:.4f}")
    print(f"    Confidence:        {res.confidence * 100:.1f}%")

    # 4. Unknown Speaker Rejection Test
    print("\n[+] Identifying Unregistered Probe Sample: 'samples/bob_test.wav'")
    res_unk = swara.identify(audio_path="samples/bob_test.wav")
    print(f"    Match Result:      {res_unk.speaker_id}")
    print(f"    Verified Match:    {res_unk.is_match}")
    print(f"    Similarity Score:  {res_unk.similarity:.4f}")

    # 5. Speaker Diarization on Multi-Speaker Conversation
    print("\n[+] Running Speaker Diarization: 'samples/conversation_test.wav'")
    diar = swara.diarize(audio_path="samples/conversation_test.wav")
    print(f"    Total Audio Duration: {diar.duration:.2f}s")
    print(f"    Detected Speakers:    {diar.num_speakers}")
    print("\n    Time-Aligned Segments:")
    for seg in diar.segments:
        ident = f" -> Identity: '{seg.identity}' ({seg.similarity:.2f})" if seg.identity else ""
        print(f"    [{seg.start:5.2f}s - {seg.end:5.2f}s] {seg.speaker_id}{ident}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
