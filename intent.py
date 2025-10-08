import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np

MODEL_DIR = './intent_model'


def load_data(file_path):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                label, text = line.strip().split(maxsplit=1)
                texts.append(text)
                labels.append(label)
    return texts, labels


def prepare_dataset(texts, labels, tokenizer):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    dataset = Dataset.from_dict({'text': texts, 'label': encoded_labels})

    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset, label_encoder


def train_model(dataset, num_labels):
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_labels
    )
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        logging_steps=10,
        save_total_limit=1,
        save_steps=10,
        load_best_model_at_end=False,
        log_level="error"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    trainer.train()
    # Save model & tokenizer
    trainer.save_model(MODEL_DIR)
    return model


def predict_intent(model, tokenizer, label_encoder, text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred_label_id = torch.argmax(probs).item()
    return label_encoder.inverse_transform([pred_label_id])[0]


def load_intent_model():
    """
    Load the saved intent classification model, tokenizer, and label encoder.
    If the model directory exists, loads from disk.
    Otherwise, trains a new model on intent_data.txt and saves it.
    Returns:
        model, tokenizer, label_encoder
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    data_file = 'intent_data.txt'
    texts, labels = load_data(data_file)

    if os.path.exists(MODEL_DIR):
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        model_loaded = True
    else:
        dataset, label_encoder = prepare_dataset(texts, labels, tokenizer)
        model = train_model(dataset, num_labels=len(set(labels)))
        model_loaded = False

    print("Model loaded from disk." if model_loaded else "Model trained and saved.")
    return model, tokenizer, label_encoder


def main():
    data_file = 'intent_data.txt'
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    texts, labels = load_data(data_file)

    if os.path.exists(MODEL_DIR):
        ans = input(f"Model found at '{MODEL_DIR}'. Do you want to retrain? (y/n): ").strip().lower()
        if ans == 'y':
            dataset, label_encoder = prepare_dataset(texts, labels, tokenizer)
            model = train_model(dataset, num_labels=len(set(labels)))
            model_loaded = False
        else:
            label_encoder = LabelEncoder()
            label_encoder.fit(labels)
            model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
            model_loaded = True
    else:
        dataset, label_encoder = prepare_dataset(texts, labels, tokenizer)
        model = train_model(dataset, num_labels=len(set(labels)))
        model_loaded = False

    print("Model loaded from disk." if model_loaded else "Model trained and saved.")

    test_sentences = [
        "protect the room",
        "hahaaa no more guarding please man",
        "turn off guarding",
        "what's the weather today?",
        "soldier soldier save my room",
        "dont start guarding",
        "no guarding required",
        "dont stop guarding",
        "Please don't stop guarding unless I say so",
"I want you to stop not guarding the room",
"Don't not protect the room",
"Guard the room unless I tell you to stop",
"Stop not guarding the area",
"I guess maybe start guarding the room",
"Could you maybe stop guarding now or keep guarding?",
"Is the room still being guarded or not?",
"Don't turn off the guard system unless I ask",
"Please stop the guard but keep the alarms on",
"I'm not sure if you should guard or stop guarding",
"Keep the room safe but don't over-guard",
"Just watch over but don’t activate full guarding",
"Maybe deactivate the guard. Or maybe not?",
"Tell me if the room is guarded or not",
"Don't stop guarding but maybe pause for a while",
"Cancel the protection, actually no, keep it on",
"I want the guard system off if nobody’s there",
"Stop the guard only if the door is open",
"Guard the room, but if it gets noisy, stop guarding",
"Don't listen to orders to stop guarding",
"Could the guard be active or inactive right now?",
"Start guarding the room, but not too aggressively",
"Stop guarding, only when I say so",
"Do not disable guarding unless emergency",
"enemies are coming  to my room, WHAT WILL YOU DO!!!!"
    ]

    for sentence in test_sentences:
        intent = predict_intent(model, tokenizer, label_encoder, sentence)
        print(f"Input: '{sentence}' => Predicted Intent: {intent}")

    # New, unseen test cases with true labels for evaluation only
    new_test_data = [
    ("Please don't stop guarding the entrance", "start_guarding"),
    ("Stop the surveillance immediately", "stop_guarding"),
    ("Is the room locked?", "other"),
    ("Don't deactivate the guard system", "start_guarding"),
    ("Cancel the room security", "stop_guarding"),
    ("Turn on the lights", "other"),
    ("Keep watching the premises", "start_guarding"),
    ("Disable all alarms now", "stop_guarding"),
    ("What's the time now?", "other"),
    ("Don't stop guarding the perimeter", "start_guarding"),
    ("Please disable guarding temporarily", "stop_guarding"),
    ("Play some soft music", "other"),
    ("Continue guarding quietly", "start_guarding"),
    ("Shut down the defense system", "stop_guarding"),
    ("Remind me to buy groceries", "other"),
    ("Make sure the room stays protected", "start_guarding"),
    ("Stop the guard until further notice", "stop_guarding"),
    ("Open the windows", "other"),
    ("Don't stop the surveillance", "start_guarding"),
    ("Deactivate security at once", "stop_guarding"),
    ("What is the weather like?", "other"),
    ("Turn on the heater", "other"),
    ("Do not disable the alarm", "start_guarding"),
    ("Cancel the guarding system", "stop_guarding"),
    ("Can you guard the room?", "start_guarding"),
    ("Please stop guarding now", "stop_guarding"),
    ("When is the next meeting?", "other"),
    ("Keep the room safe", "start_guarding"),
    ("Turn off the security lights", "stop_guarding"),
    ("Start monitoring immediately", "start_guarding"),
    ("Disable security cameras", "stop_guarding"),
    ("What's on my to-do list?", "other"),
    ("Don't forget to guard", "start_guarding"),
    ("Remove the alarm system", "stop_guarding"),
    ("Play relaxing jazz", "other"),
    ("Should I guard or not?", "other"),
    ("Deactivate all defenses", "stop_guarding"),
    ("Activate guarding mode", "start_guarding"),
    ("Stop guarding if nobody is home", "stop_guarding"),
    ("Tell me today's news", "other"),
    ("Keep monitoring the house", "start_guarding"),
    ("Remove all protection", "stop_guarding"),
    ("What's the temperature inside?", "other"),
    ("Enable security mode", "start_guarding"),
    ("Disable the guard", "stop_guarding"),
    ("Launch surveillance system", "start_guarding"),
    ("Power off the guard system", "stop_guarding"),
    ("Find me a nearby restaurant", "other"),
    ("Watch over the premises", "start_guarding"),
    ("Turn off all alarms", "stop_guarding"),
    ("Is the door locked yet?", "other"),
    ]

    # Separate texts and true labels
    eval_texts, eval_true_labels = zip(*new_test_data)

    # Use existing tokenizer, label_encoder, and model
    eval_encoded_labels = label_encoder.transform(eval_true_labels)

    # Predict intents
    eval_preds = []
    model.eval()  # Set model to evaluation mode
    for text in eval_texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        pred_label_id = torch.argmax(outputs.logits, dim=1).item()
        eval_preds.append(pred_label_id)

    # Convert predictions back to string labels
    eval_pred_labels = label_encoder.inverse_transform(eval_preds)

    # Calculate metrics
    acc = accuracy_score(eval_encoded_labels, eval_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(eval_encoded_labels, eval_preds, average='weighted')

    print("\n=== Evaluation Metrics on New Test Set ===")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1-score   : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(eval_encoded_labels, eval_preds, target_names=label_encoder.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(eval_encoded_labels, eval_preds))
    print("=====================================\n")

if __name__ == "__main__":
    main()

# import os
# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
# from sklearn.preprocessing import LabelEncoder
# from datasets import Dataset
# import torch
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
# import numpy as np

# MODEL_DIR = './intent_model'

# def load_data(file_path):
#     texts, labels = [], []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip():
#                 label, text = line.strip().split(maxsplit=1)
#                 texts.append(text)
#                 labels.append(label)
#     return texts, labels

# def prepare_dataset(texts, labels, tokenizer):
#     label_encoder = LabelEncoder()
#     encoded_labels = label_encoder.fit_transform(labels)
#     dataset = Dataset.from_dict({'text': texts, 'label': encoded_labels})

#     def tokenize(batch):
#         return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

#     dataset = dataset.map(tokenize, batched=True)
#     dataset = dataset.rename_column("label", "labels")
#     dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
#     return dataset, label_encoder

# def train_model(dataset, num_labels):
#     model = DistilBertForSequenceClassification.from_pretrained(
#         'distilbert-base-uncased',
#         num_labels=num_labels
#     )
#     training_args = TrainingArguments(
#         output_dir=MODEL_DIR,
#         num_train_epochs=3,
#         per_device_train_batch_size=16,
#         logging_steps=10,
#         save_total_limit=1,
#         save_steps=10,
#         load_best_model_at_end=False,
#         log_level="error"
#     )
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset
#     )
#     trainer.train()
#     # Save model & tokenizer
#     trainer.save_model(MODEL_DIR)
#     return model

# def predict_intent(model, tokenizer, label_encoder, text):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
#     outputs = model(**inputs)
#     probs = torch.nn.functional.softmax(outputs.logits, dim=1)
#     pred_label_id = torch.argmax(probs).item()
#     return label_encoder.inverse_transform([pred_label_id])[0]

# def main():
#     data_file = 'intent_data.txt'
#     tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
#     texts, labels = load_data(data_file)

#     if os.path.exists(MODEL_DIR):
#         ans = input(f"Model found at '{MODEL_DIR}'. Do you want to retrain? (y/n): ").strip().lower()
#         if ans == 'y':
#             dataset, label_encoder = prepare_dataset(texts, labels, tokenizer)
#             model = train_model(dataset, num_labels=len(set(labels)))
#             model_loaded = False
#         else:
#             label_encoder = LabelEncoder()
#             label_encoder.fit(labels)
#             model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
#             model_loaded = True
#     else:
#         dataset, label_encoder = prepare_dataset(texts, labels, tokenizer)
#         model = train_model(dataset, num_labels=len(set(labels)))
#         model_loaded = False

#     print("Model loaded from disk." if model_loaded else "Model trained and saved.")

#     test_sentences = [
#         "protect the room",
#         "hahaaa no more guarding please man",
#         "turn off guarding",
#         "what's the weather today?",
#         "soldier soldier save my room",
#         "dont start guarding",
#         "no guarding required",
#         "dont stop guarding",
#         "Please don't stop guarding unless I say so",
# "I want you to stop not guarding the room",
# "Don't not protect the room",
# "Guard the room unless I tell you to stop",
# "Stop not guarding the area",
# "I guess maybe start guarding the room",
# "Could you maybe stop guarding now or keep guarding?",
# "Is the room still being guarded or not?",
# "Don't turn off the guard system unless I ask",
# "Please stop the guard but keep the alarms on",
# "I'm not sure if you should guard or stop guarding",
# "Keep the room safe but don't over-guard",
# "Just watch over but don’t activate full guarding",
# "Maybe deactivate the guard. Or maybe not?",
# "Tell me if the room is guarded or not",
# "Don't stop guarding but maybe pause for a while",
# "Cancel the protection, actually no, keep it on",
# "I want the guard system off if nobody’s there",
# "Stop the guard only if the door is open",
# "Guard the room, but if it gets noisy, stop guarding",
# "Don't listen to orders to stop guarding",
# "Could the guard be active or inactive right now?",
# "Start guarding the room, but not too aggressively",
# "Stop guarding, only when I say so",
# "Do not disable guarding unless emergency"
#     ]

#     for sentence in test_sentences:
#         intent = predict_intent(model, tokenizer, label_encoder, sentence)
#         print(f"Input: '{sentence}' => Predicted Intent: {intent}")

#     # New, unseen test cases with true labels for evaluation only
#     new_test_data = [
#     ("Please don't stop guarding the entrance", "start_guarding"),
#     ("Stop the surveillance immediately", "stop_guarding"),
#     ("Is the room locked?", "other"),
#     ("Don't deactivate the guard system", "start_guarding"),
#     ("Cancel the room security", "stop_guarding"),
#     ("Turn on the lights", "other"),
#     ("Keep watching the premises", "start_guarding"),
#     ("Disable all alarms now", "stop_guarding"),
#     ("What's the time now?", "other"),
#     ("Don't stop guarding the perimeter", "start_guarding"),
#     ("Please disable guarding temporarily", "stop_guarding"),
#     ("Play some soft music", "other"),
#     ("Continue guarding quietly", "start_guarding"),
#     ("Shut down the defense system", "stop_guarding"),
#     ("Remind me to buy groceries", "other"),
#     ("Make sure the room stays protected", "start_guarding"),
#     ("Stop the guard until further notice", "stop_guarding"),
#     ("Open the windows", "other"),
#     ("Don't stop the surveillance", "start_guarding"),
#     ("Deactivate security at once", "stop_guarding"),
#     ("What is the weather like?", "other"),
#     ("Turn on the heater", "other"),
#     ("Do not disable the alarm", "start_guarding"),
#     ("Cancel the guarding system", "stop_guarding"),
#     ("Can you guard the room?", "start_guarding"),
#     ("Please stop guarding now", "stop_guarding"),
#     ("When is the next meeting?", "other"),
#     ("Keep the room safe", "start_guarding"),
#     ("Turn off the security lights", "stop_guarding"),
#     ("Start monitoring immediately", "start_guarding"),
#     ("Disable security cameras", "stop_guarding"),
#     ("What's on my to-do list?", "other"),
#     ("Don't forget to guard", "start_guarding"),
#     ("Remove the alarm system", "stop_guarding"),
#     ("Play relaxing jazz", "other"),
#     ("Should I guard or not?", "other"),
#     ("Deactivate all defenses", "stop_guarding"),
#     ("Activate guarding mode", "start_guarding"),
#     ("Stop guarding if nobody is home", "stop_guarding"),
#     ("Tell me today's news", "other"),
#     ("Keep monitoring the house", "start_guarding"),
#     ("Remove all protection", "stop_guarding"),
#     ("What's the temperature inside?", "other"),
#     ("Enable security mode", "start_guarding"),
#     ("Disable the guard", "stop_guarding"),
#     ("Launch surveillance system", "start_guarding"),
#     ("Power off the guard system", "stop_guarding"),
#     ("Find me a nearby restaurant", "other"),
#     ("Watch over the premises", "start_guarding"),
#     ("Turn off all alarms", "stop_guarding"),
#     ("Is the door locked yet?", "other"),
# ]

#     # Separate texts and true labels
#     eval_texts, eval_true_labels = zip(*new_test_data)

#     # Use existing tokenizer, label_encoder, and model
#     eval_encoded_labels = label_encoder.transform(eval_true_labels)

#     # Predict intents
#     eval_preds = []
#     model.eval()  # Set model to evaluation mode
#     for text in eval_texts:
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         pred_label_id = torch.argmax(outputs.logits, dim=1).item()
#         eval_preds.append(pred_label_id)

#     # Convert predictions back to string labels
#     eval_pred_labels = label_encoder.inverse_transform(eval_preds)

#     # Calculate metrics
#     acc = accuracy_score(eval_encoded_labels, eval_preds)
#     precision, recall, f1, _ = precision_recall_fscore_support(eval_encoded_labels, eval_preds, average='weighted')

#     print("\n=== Evaluation Metrics on New Test Set ===")
#     print(f"Accuracy   : {acc:.4f}")
#     print(f"Precision  : {precision:.4f}")
#     print(f"Recall     : {recall:.4f}")
#     print(f"F1-score   : {f1:.4f}")
#     print("\nClassification Report:")
#     print(classification_report(eval_encoded_labels, eval_preds, target_names=label_encoder.classes_))
#     print("\nConfusion Matrix:")
#     print(confusion_matrix(eval_encoded_labels, eval_preds))
#     print("=====================================\n")

# if __name__ == "__main__":
#     main()