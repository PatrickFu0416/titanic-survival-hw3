"""Train, evaluate, and run a Titanic logistic regression from the terminal.

Run from the repository root: python src/python/main.py
Input CSVs remain unchanged. All learned preprocessing is fitted on training rows.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = ['Age', 'Fare', 'FamilySize']
CATEGORICAL = ['Sex', 'Pclass', 'Embarked']
REQUIRED = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


def read_passengers(path, training=False):
    """Load and check the input schema before constructing any features."""
    if not path.is_file():
        raise ValueError(f'Missing {path}. Download train.csv and test.csv into src/data/.')
    frame = pd.read_csv(path)
    missing = set(REQUIRED + (['Survived'] if training else [])) - set(frame.columns)
    if missing:
        raise ValueError(f'{path.name} is missing columns: {sorted(missing)}')
    if frame.empty or frame['PassengerId'].isna().any() or not frame['PassengerId'].is_unique:
        raise ValueError(f'{path.name} must contain rows with unique, nonmissing PassengerId values.')
    for column in ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if not np.isfinite(frame[column].dropna()).all():
            raise ValueError(f'{column} contains infinite values.')
    for column in ['PassengerId', 'Pclass', 'SibSp', 'Parch']:
        values = frame[column]
        if values.isna().any() or not values.eq(np.floor(values)).all():
            raise ValueError(f'{column} must contain nonmissing integers.')
    if (frame['PassengerId'] <= 0).any():
        raise ValueError('PassengerId must be positive.')
    if (frame[['Age', 'Fare', 'SibSp', 'Parch']] < 0).any().any():
        raise ValueError('Age, Fare, SibSp, and Parch cannot be negative.')
    for column in ['Sex', 'Embarked']:
        frame[column] = frame[column].astype(object).map(
            lambda value: value.strip() if isinstance(value, str) else value
        ).replace('', np.nan)
    if not frame['Sex'].isin(['female', 'male']).all():
        raise ValueError('Sex must be female or male in the supplied competition data.')
    if not frame['Pclass'].isin([1, 2, 3]).all():
        raise ValueError('Pclass must be 1, 2, or 3.')
    if not frame['Embarked'].dropna().isin(['C', 'Q', 'S']).all():
        raise ValueError('Embarked must be C, Q, S, or missing.')
    if training:
        if not frame['Survived'].isin([0, 1]).all() or frame['Survived'].nunique() != 2:
            raise ValueError('Training Survived must contain both 0 and 1, with no missing labels.')
    print(f'Loaded {path.name}: {len(frame)} rows, {len(frame.columns)} columns.')
    return frame


def features(frame):
    """Keep selected predictors and add family size without using survival labels."""
    result = frame[['Age', 'Fare', 'Sex', 'Pclass', 'Embarked']].copy()
    result['FamilySize'] = frame['SibSp'] + frame['Parch'] + 1
    return result[NUMERIC + CATEGORICAL]


def build_model():
    """A fitted pipeline keeps imputation, encoding, and scaling leakage-safe."""
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), NUMERIC),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore')),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ('preprocess', preprocessing),
        ('model', LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)),
    ])


def fit_model(x, y, label):
    """Print the preprocessing values learned for this particular fit."""
    model = build_model().fit(x, y)
    transform = model.named_steps['preprocess']
    medians = transform.named_transformers_['numeric'].named_steps['impute'].statistics_
    modes = transform.named_transformers_['categorical'].named_steps['impute'].statistics_
    print(f'{label}: training-only numeric medians: {dict(zip(NUMERIC, medians.tolist()))}')
    print(f'{label}: training-only categorical modes: {dict(zip(CATEGORICAL, modes.tolist()))}')
    return model


def evaluate_test_labels(path, test, predictions):
    """Score only separately supplied genuine labels, joined by PassengerId."""
    labels = pd.read_csv(path)
    if set(labels.columns) != {'PassengerId', 'Survived'}:
        raise ValueError('Test-label file must have exactly PassengerId,Survived columns.')
    if (labels['PassengerId'].isna().any() or not labels['PassengerId'].is_unique
            or set(labels['PassengerId']) != set(test['PassengerId'])):
        raise ValueError('Test labels must cover every test PassengerId exactly once.')
    if not labels['Survived'].isin([0, 1]).all():
        raise ValueError('Test labels must be genuine, nonmissing 0/1 outcomes.')
    aligned = test[['PassengerId']].merge(labels, on='PassengerId', validate='one_to_one')
    return float(accuracy_score(aligned['Survived'], predictions))


def run(args):
    """Print exploration, transformations and metrics, then save predictions."""
    print('Titanic survival | Python | logistic regression')
    train = read_passengers(args.data_dir / 'train.csv', training=True)
    test = read_passengers(args.data_dir / 'test.csv')
    if set(train['PassengerId']) & set(test['PassengerId']):
        raise ValueError('Training and competition-test PassengerId values must not overlap.')
    print('\nQ14 - Explore and transform')
    print('Training survival by sex (count and mean):')
    print(train.groupby('Sex')['Survived'].agg(['count', 'mean']).to_string())
    print('Training survival by passenger class:')
    print(train.groupby('Pclass')['Survived'].agg(['count', 'mean']).to_string())
    for label, frame in [('train', train), ('test', test)]:
        print(f'{label} missing predictor values: {frame[REQUIRED].isna().sum().to_dict()}')
    print('Keep every passenger; trim categorical text; add FamilySize = SibSp + Parch + 1.')
    print('Use Age, Fare, FamilySize, Sex, Pclass, Embarked; omit IDs and high-cardinality text/Cabin.')
    print('Median-impute numeric values; mode-impute categories; scale numeric features; one-hot encode categories.')
    print('Fit preprocessing on training rows only; never use test data to estimate imputations.')
    x, y = features(train), train['Survived'].astype(int)
    fit_x, valid_x, fit_y, valid_y = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f'\nQ15 - Fixed 80/20 stratified validation split: {len(fit_x)} fit / {len(valid_x)} holdout.')
    validation_model = fit_model(fit_x, fit_y, 'Validation fit')
    valid_predictions = validation_model.predict(valid_x)
    validation_accuracy = float(accuracy_score(valid_y, valid_predictions))
    baseline_accuracy = float(accuracy_score(valid_y, np.repeat(fit_y.mode().iloc[0], len(valid_y))))
    print(f'Held-out validation accuracy: {validation_accuracy:.6f} ({validation_accuracy:.2%})')
    print(f'Majority-class validation baseline: {baseline_accuracy:.6f}')
    print('Validation confusion matrix: rows=true 0/1, columns=predicted 0/1')
    print(confusion_matrix(valid_y, valid_predictions, labels=[0, 1]))
    print('\nQ16 - Refit preprocessing and logistic regression on all labeled training rows.')
    model = fit_model(x, y, 'Full training fit')
    training_accuracy = float(accuracy_score(y, model.predict(x)))
    print(f'Training accuracy (resubstitution): {training_accuracy:.6f} ({training_accuracy:.2%})')
    print('\nQ17 - Predict the separate competition test set using the full-training fit.')
    probabilities = model.predict_proba(features(test))[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions}).to_csv(
        args.output_dir / 'predictions.csv', index=False
    )
    pd.DataFrame({'PassengerId': test['PassengerId'], 'SurvivalProbability': probabilities}).to_csv(
        args.output_dir / 'probabilities.csv', index=False
    )
    print(f'Wrote {len(predictions)} predictions; class counts: {pd.Series(predictions).value_counts().sort_index().to_dict()}')
    print('\nQ18 - Competition test accuracy')
    test_accuracy = None
    if args.test_labels is not None:
        test_accuracy = evaluate_test_labels(args.test_labels, test, predictions)
        print(f'Test accuracy against supplied genuine labels: {test_accuracy:.6f}')
    else:
        print('NOT AVAILABLE: Kaggle test.csv does not provide ground-truth Survived labels.')
        print('The holdout accuracy above is an honest unseen-labeled-data estimate, not a Kaggle test score.')
        print('gender_submission.csv is an example baseline, NOT ground truth. Do not score against it.')
    pd.DataFrame([
        {'metric': 'training_accuracy', 'value': training_accuracy, 'rows': len(train)},
        {'metric': 'holdout_accuracy', 'value': validation_accuracy, 'rows': len(valid_y)},
        {'metric': 'holdout_majority_baseline', 'value': baseline_accuracy, 'rows': len(valid_y)},
        {'metric': 'competition_test_accuracy', 'value': test_accuracy, 'rows': len(test)},
    ]).to_csv(args.output_dir / 'metrics.csv', index=False)
    print(f'Completed successfully. Results: {args.output_dir.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('src/data'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/python'))
    parser.add_argument('--test-labels', type=Path, help='Optional genuine test outcomes, never gender_submission.csv.')
    arguments = parser.parse_args()
    try:
        run(arguments)
    except (ValueError, OSError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
