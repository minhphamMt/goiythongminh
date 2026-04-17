class EmbeddingServiceError(Exception):
    retryable = True


class NonRetryableEmbeddingError(EmbeddingServiceError):
    retryable = False


class SongNotFoundError(NonRetryableEmbeddingError):
    pass


class MetadataNotFoundError(NonRetryableEmbeddingError):
    pass


class MissingAudioPathError(NonRetryableEmbeddingError):
    pass


class AudioDecodeError(NonRetryableEmbeddingError):
    pass
