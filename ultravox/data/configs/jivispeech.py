from ultravox.data import types






JIVI_HI_STT_CONFIG = types.DatasetConfig(
    name="/home/akshat/speech-experiments/data_storage_volume/final_data/asr_hi_jivi_train",
    # base="",
    path= "/home/akshat/speech-experiments/data_storage_volume/final_data/asr_hi_jivi_train",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    transcript_template="{{sentence}}",
    assistant_template="{{sentence}}",
    splits=[
        types.DatasetSplitConfig(name="nosplit", num_samples=960_033),
    ],
)


JIVI_HI_TEST_STT_CONFIG = types.DatasetConfig(
    name="/home/akshat/speech-experiments/data_storage_volume/final_data/asr_hi_jivi_test",
    # base="",
    path= "/home/akshat/speech-experiments/data_storage_volume/final_data/asr_hi_jivi_test",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    transcript_template="{{sentence}}",
    assistant_template="{{sentence}}",
    splits=[
        types.DatasetSplitConfig(name="nosplit", num_samples=50_000),
    ],
)





HANI_MEDICAL = types.DatasetConfig(
    name="Hani89/medical_asr_recording_dataset",
    path="Hani89/medical_asr_recording_dataset",
    subset=None,
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=5330, split_type=types.DatasetSplit.TRAIN
        )
    ],
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    transcript_template="{{sentence}}",
    assistant_template="{{sentence}}",
)

JARVIS_MEDICAL = types.DatasetConfig(
    name="jarvisx17/Medical-ASR-EN",
    path="jarvisx17/Medical-ASR-EN",
    subset=None,
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=5640, split_type=types.DatasetSplit.TRAIN
        )
    ],
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    transcript_template="{{transcription}}",
    assistant_template="{{transcription}}",
)

configs = [
    JIVI_HI_STT_CONFIG,
    JIVI_HI_TEST_STT_CONFIG,
    HANI_MEDICAL,
    JARVIS_MEDICAL,
]
