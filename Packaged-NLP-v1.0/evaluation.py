def evaluate_model(model, eval_questions, eval_answers, batch_size):
    """
    Evaluates the model on the evaluation data and returns the prediction accuracy.

    Args:
        model: The trained model.
        eval_questions: The padded sequences of tokenized evaluation questions.
        eval_answers: The one-hot encoded evaluation answers.
        batch_size: The batch size for evaluation.

    Returns:
        accuracy: The prediction accuracy of the model on the evaluation data.
    """
    _, accuracy = model.evaluate(eval_questions, eval_answers, batch_size=batch_size)
    print("Prediction accuracy:", accuracy)
    