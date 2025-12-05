# Tokenize data
def tokenize_data(texts, labels, max_len=128):
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            max_length=max_len,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='tf'
        )
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
        token_type_ids.append(encoding['token_type_ids'])

    return {
        'input_ids': tf.concat(input_ids, axis=0),
        'attention_mask': tf.concat(attention_masks, axis=0),
        'token_type_ids': tf.concat(token_type_ids, axis=0)
    }, tf.convert_to_tensor(labels, dtype=tf.float32)

train_data, train_labels = tokenize_data(X_train, y_train)
val_data, val_labels = tokenize_data(X_val, y_val)
test_data, test_labels = tokenize_data(X_test, y_test)

