-- 在 SQLite 数据库工具中打开 artifacts/tiny-inference/token-embeddings.sqlite3。
-- 每次选择一段 SQL 执行。

-- 1. 词表前 30 项。escaped 列用引号和转义符标出空格、换行。
SELECT token_id, token_text, token_text_escaped, token_kind
FROM tokens ORDER BY token_id LIMIT 30;

-- 2. 9038 是 " Once"，938 是 " int"；二者不同。
SELECT token_id, token_text_escaped, embedding_000, embedding_001, embedding_002
FROM tokens WHERE token_id IN (938, 9038);

-- 3. 纵向查看 Once 的全部 288 个数，dim_index 从 0 到 287。
SELECT * FROM once_embedding;

-- 4. 换成任意 token ID，都可以逐维查看。
SELECT dim_index, value FROM embedding_values
WHERE token_id = 2501 ORDER BY dim_index;

-- 5. 按文字查 token。带空格与不带空格可能是不同 token。
SELECT token_id, token_text_escaped FROM tokens
WHERE token_text IN ('Once', ' Once');

-- 6. 横向查看一个 token，embedding_000 到 embedding_287 对应数组下标。
SELECT * FROM tokens WHERE token_id = 9038;

-- 7. 查看模型配置、哈希和源码版本。
SELECT * FROM metadata;
