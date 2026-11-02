# Titanic logistic regression using only R's bundled stats and utils packages.
# From the repository root: Rscript src/r/main.R

numeric_features <- c("Age", "Fare", "FamilySize")
categorical_features <- c("Sex", "Pclass", "Embarked")
required_columns <- c("PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch",
                      "Fare", "Embarked")
category_levels <- list(Sex = c("female", "male"), Pclass = c("1", "2", "3"),
                        Embarked = c("C", "Q", "S"))

read_passengers <- function(path, training = FALSE) {
  # Validate the schema before deriving features. Keep the source files unchanged.
  if (!file.exists(path)) {
    stop(sprintf("Missing %s. Download train.csv and test.csv into src/data/.", path))
  }
  frame <- utils::read.csv(path, stringsAsFactors = FALSE, na.strings = c("", "NA"))
  needed <- c(required_columns, if (training) "Survived")
  missing <- setdiff(needed, names(frame))
  if (length(missing) > 0) {
    stop(sprintf("%s is missing columns: %s", basename(path), paste(missing, collapse = ", ")))
  }
  if (nrow(frame) == 0 || anyNA(frame$PassengerId) || anyDuplicated(frame$PassengerId)) {
    stop("Input must contain rows with unique, nonmissing PassengerId values.")
  }
  for (column in c("PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare")) {
    old <- frame[[column]]
    values <- suppressWarnings(as.numeric(old))
    if (any(is.na(values) & !is.na(old)) || any(!is.finite(values[!is.na(values)]))) {
      stop(sprintf("%s contains invalid or infinite numeric values.", column))
    }
    frame[[column]] <- values
  }
  for (column in c("PassengerId", "Pclass", "SibSp", "Parch")) {
    values <- frame[[column]]
    if (anyNA(values) || any(values != floor(values))) {
      stop(sprintf("%s must contain nonmissing integers.", column))
    }
  }
  if (any(frame$PassengerId <= 0) || anyDuplicated(frame$PassengerId)) {
    stop("PassengerId must be positive and unique.")
  }
  if (any(frame[c("Age", "Fare", "SibSp", "Parch")] < 0, na.rm = TRUE)) {
    stop("Age, Fare, SibSp, and Parch cannot be negative.")
  }
  for (column in c("Sex", "Embarked")) {
    frame[[column]] <- trimws(frame[[column]])
    frame[[column]][!is.na(frame[[column]]) & frame[[column]] == ""] <- NA_character_
  }
  if (!all(frame$Sex %in% category_levels$Sex)) stop("Sex must be female or male.")
  if (!all(frame$Pclass %in% c(1, 2, 3))) stop("Pclass must be 1, 2, or 3.")
  if (!all(stats::na.omit(frame$Embarked) %in% category_levels$Embarked)) {
    stop("Embarked must be C, Q, S, or missing.")
  }
  if (training && (!all(frame$Survived %in% c(0, 1)) ||
                   length(unique(frame$Survived)) != 2)) {
    stop("Training Survived must contain both 0 and 1, with no missing labels.")
  }
  cat(sprintf("Loaded %s: %d rows, %d columns.\n", basename(path), nrow(frame), ncol(frame)))
  frame
}

features <- function(frame) {
  # IDs and survival labels never enter the predictor table.
  result <- frame[c("Age", "Fare", "Sex", "Pclass", "Embarked")]
  result$FamilySize <- frame$SibSp + frame$Parch + 1
  result$Pclass <- as.character(result$Pclass)
  result[c(numeric_features, categorical_features)]
}

mode_value <- function(values) {
  counts <- table(values)
  if (length(counts) == 0) stop("A categorical training column is entirely missing.")
  names(counts)[which.max(counts)]
}

learn_preprocessing <- function(x) {
  # All medians and modes come from fitting rows only, never holdout/test rows.
  medians <- vapply(x[numeric_features], stats::median, numeric(1), na.rm = TRUE)
  if (any(!is.finite(medians))) stop("A numeric training column is entirely missing.")
  modes <- vapply(x[categorical_features], mode_value, character(1))
  list(medians = medians, modes = modes)
}

apply_preprocessing <- function(x, settings) {
  for (column in numeric_features) {
    x[[column]][is.na(x[[column]])] <- settings$medians[[column]]
  }
  for (column in categorical_features) {
    x[[column]][is.na(x[[column]])] <- settings$modes[[column]]
    # These fixed levels come from the data dictionary, not from test observations.
    x[[column]] <- factor(x[[column]], levels = category_levels[[column]])
    if (anyNA(x[[column]])) stop(sprintf("Unknown category in %s.", column))
  }
  x
}

fit_model <- function(x, y, label) {
  settings <- learn_preprocessing(x)
  cat(label, "- training-only medians:\n")
  print(settings$medians)
  cat(label, "- training-only categorical modes:\n")
  print(settings$modes)
  prepared <- apply_preprocessing(x, settings)
  prepared$Survived <- y
  model <- stats::glm(
    Survived ~ Age + Fare + FamilySize + Sex + Pclass + Embarked,
    data = prepared, family = stats::binomial(link = "logit"),
    control = stats::glm.control(maxit = 100)
  )
  if (!model$converged) stop("Logistic regression did not converge.")
  list(model = model, preprocessing = settings)
}

predict_probabilities <- function(fit, x) {
  result <- stats::predict(fit$model, newdata = apply_preprocessing(x, fit$preprocessing),
                           type = "response")
  if (anyNA(result) || any(!is.finite(result))) stop("Prediction produced invalid probabilities.")
  as.numeric(result)
}

evaluate_test_labels <- function(path, test, predictions) {
  # Optional genuine outcomes are joined by ID, independently of their row order.
  labels <- utils::read.csv(path)
  if (!setequal(names(labels), c("PassengerId", "Survived")) || ncol(labels) != 2) {
    stop("Test-label file must have exactly PassengerId,Survived columns.")
  }
  if (anyNA(labels$PassengerId) || anyDuplicated(labels$PassengerId) ||
      !setequal(labels$PassengerId, test$PassengerId)) {
    stop("Test labels must cover every test PassengerId exactly once.")
  }
  if (!all(labels$Survived %in% c(0, 1))) stop("Test labels must be nonmissing 0/1 outcomes.")
  aligned <- labels$Survived[match(test$PassengerId, labels$PassengerId)]
  mean(aligned == predictions)
}

run <- function(args) {
  cat("Titanic survival | R | binomial logistic regression\n")
  train <- read_passengers(file.path(args$data_dir, "train.csv"), training = TRUE)
  test <- read_passengers(file.path(args$data_dir, "test.csv"))
  if (length(intersect(train$PassengerId, test$PassengerId)) > 0) {
    stop("Training and competition-test PassengerId values must not overlap.")
  }
  cat("\nQ14 - Explore and transform\n")
  cat("Training survival counts by sex:\n")
  print(table(train$Sex, train$Survived))
  cat("Training survival rates by sex and class:\n")
  print(stats::aggregate(Survived ~ Sex, train, mean))
  print(stats::aggregate(Survived ~ Pclass, train, mean))
  cat("Training missing predictor values:\n")
  print(colSums(is.na(train[required_columns])))
  cat("Test missing predictor values:\n")
  print(colSums(is.na(test[required_columns])))
  cat("Keep every passenger; trim categorical text; add FamilySize = SibSp + Parch + 1.\n")
  cat("Use Age, Fare, FamilySize, Sex, Pclass, Embarked; omit IDs and text/Cabin.\n")
  cat("Median-impute numeric values and mode-impute categories using fitting rows only.\n")
  cat("Use factors for categorical predictors and an unpenalized binomial glm.\n")
  x <- features(train)
  y <- train$Survived
  set.seed(42)
  fitting <- unlist(lapply(c(0, 1), function(label) {
    candidates <- which(y == label)
    if (length(candidates) < 2) stop("Each outcome needs at least two rows for validation.")
    sample(candidates, size = floor(0.8 * length(candidates)))
  }))
  holdout <- setdiff(seq_len(nrow(train)), fitting)
  cat(sprintf("\nQ15 - Seed-42 stratified split: %d fit / %d holdout.\n",
              length(fitting), length(holdout)))
  validation_model <- fit_model(x[fitting, ], y[fitting], "Validation fit")
  valid_predictions <- as.integer(predict_probabilities(validation_model, x[holdout, ]) >= 0.5)
  validation_accuracy <- mean(valid_predictions == y[holdout])
  baseline <- as.numeric(mode_value(y[fitting]))
  baseline_accuracy <- mean(y[holdout] == baseline)
  cat(sprintf("Held-out validation accuracy: %.6f (%.2f%%)\n",
              validation_accuracy, 100 * validation_accuracy))
  cat(sprintf("Majority-class validation baseline: %.6f\n", baseline_accuracy))
  cat("Validation confusion matrix: rows=true 0/1, columns=predicted 0/1\n")
  print(table(True = factor(y[holdout], levels = c(0, 1)),
              Predicted = factor(valid_predictions, levels = c(0, 1))))
  cat("\nQ16 - Refit preprocessing and logistic regression on all labeled training rows.\n")
  model <- fit_model(x, y, "Full training fit")
  training_accuracy <- mean(as.integer(predict_probabilities(model, x) >= 0.5) == y)
  cat(sprintf("Training accuracy (resubstitution): %.6f (%.2f%%)\n",
              training_accuracy, 100 * training_accuracy))
  cat("\nQ17 - Predict the separate competition test set using the full-training fit.\n")
  probabilities <- predict_probabilities(model, features(test))
  predictions <- as.integer(probabilities >= 0.5)
  dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(data.frame(PassengerId = test$PassengerId, Survived = predictions),
                   file.path(args$output_dir, "predictions.csv"), row.names = FALSE)
  utils::write.csv(data.frame(PassengerId = test$PassengerId,
                              SurvivalProbability = probabilities),
                   file.path(args$output_dir, "probabilities.csv"), row.names = FALSE)
  cat(sprintf("Wrote %d test predictions; class counts:\n", length(predictions)))
  print(table(predictions))
  cat("\nQ18 - Competition test accuracy\n")
  test_accuracy <- NA_real_
  if (!is.null(args$test_labels)) {
    test_accuracy <- evaluate_test_labels(args$test_labels, test, predictions)
    cat(sprintf("Test accuracy against supplied genuine labels: %.6f\n", test_accuracy))
  } else {
    cat("NOT AVAILABLE: Kaggle test.csv does not provide ground-truth Survived labels.\n")
    cat("The holdout score is an unseen-labeled-data estimate, not a Kaggle test score.\n")
    cat("gender_submission.csv is a baseline example, NOT ground truth.\n")
  }
  metrics <- data.frame(
    metric = c("training_accuracy", "holdout_accuracy", "holdout_majority_baseline",
               "competition_test_accuracy"),
    value = c(training_accuracy, validation_accuracy, baseline_accuracy, test_accuracy),
    rows = c(nrow(train), length(holdout), length(holdout), nrow(test))
  )
  utils::write.csv(metrics, file.path(args$output_dir, "metrics.csv"),
                   row.names = FALSE, na = "")
  cat("Completed successfully. Results:", normalizePath(args$output_dir), "\n")
}

parse_arguments <- function(values) {
  if (length(values) == 1 && values[1] == "--help") {
    cat("Rscript src/r/main.R [--data-dir DIR] [--output-dir DIR] [--test-labels FILE]\n")
    quit(status = 0)
  }
  if (length(values) %% 2 != 0) stop("Options require values. Use --help for usage.")
  result <- list(data_dir = "src/data", output_dir = "outputs/r", test_labels = NULL)
  if (length(values) > 0) {
    for (index in seq(1, length(values), by = 2)) {
      key <- values[index]
      if (!(key %in% c("--data-dir", "--output-dir", "--test-labels"))) {
        stop(sprintf("Unknown option: %s", key))
      }
      result[[gsub("-", "_", substring(key, 3))]] <- values[index + 1]
    }
  }
  result
}

if (sys.nframe() == 0) {
  tryCatch(run(parse_arguments(commandArgs(trailingOnly = TRUE))), error = function(error) {
    cat("ERROR:", conditionMessage(error), "\n", file = stderr())
    quit(status = 1)
  })
}
