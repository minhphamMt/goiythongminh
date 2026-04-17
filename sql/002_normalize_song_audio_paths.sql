USE music_platform;

-- Normalize full Firebase download URLs into blob-style paths.
UPDATE songs
SET audio_path = SUBSTRING_INDEX(audio_path, '/o/', -1)
WHERE audio_path LIKE '%/o/%';

-- Remove query strings such as ?alt=media&token=...
UPDATE songs
SET audio_path = SUBSTRING_INDEX(audio_path, '?', 1)
WHERE audio_path LIKE '%?%';

-- Decode double-encoded forward slashes first.
UPDATE songs
SET audio_path = REPLACE(audio_path, '%252F', '%2F')
WHERE audio_path LIKE '%252F%';

-- Decode encoded forward slashes.
UPDATE songs
SET audio_path = REPLACE(audio_path, '%2F', '/')
WHERE audio_path LIKE '%2F%';

-- Remove malformed rows that still contain a nested Firebase URL
-- after a leading uploads/ prefix.
UPDATE songs
SET audio_path = REPLACE(
    audio_path,
    'uploads/https://firebasestorage.googleapis.com/v0/b/khoaluantotnghiep-bc862.firebasestorage.app/o/',
    ''
)
WHERE audio_path LIKE 'uploads/https://firebasestorage.googleapis.com/v0/b/%/o/%';

-- Collapse duplicated uploads prefixes.
UPDATE songs
SET audio_path = REPLACE(audio_path, 'uploads/uploads/', 'uploads/')
WHERE audio_path LIKE 'uploads/uploads/%';

-- Ensure paths rooted at uploads keep a single leading prefix.
UPDATE songs
SET audio_path = CONCAT('uploads/', TRIM(LEADING '/' FROM audio_path))
WHERE audio_path IS NOT NULL
  AND TRIM(audio_path) <> ''
  AND audio_path NOT LIKE 'uploads/%';

