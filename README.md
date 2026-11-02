# Titanic Survival: Python and R with Docker

MLDS 400 Homework 3 - Hongcheng Fu.

Clone this repository, download two Kaggle files, and run a logistic regression
pipeline from your terminal. Each pipeline prints its data checks, cleaning,
feature choices, training accuracy, holdout accuracy, and prediction summary.
The input dataset is kept outside Git and is mounted into containers at runtime.

## 1. Install prerequisites

- Install [Git](https://git-scm.com/downloads).
- Install and start [Docker Desktop](https://www.docker.com/products/docker-desktop/).
  On Linux, Docker Engine also works. On macOS, Docker CLI with Colima also works.
- Use Terminal on macOS/Linux, or **Git Bash** on Windows for the commands below.
  With Docker Desktop on Windows, select Linux containers.
- A Kaggle account is needed to download the competition data directly.

Check that the Docker **server** is running:

```bash
git --version
docker info
```

Installing Python or R on your computer is optional: both runtimes and all
modeling dependencies are provided by the Docker images. The first build needs
internet access to download the base images and Python dependencies. Later builds
reuse the cache. Prediction itself needs no internet connection.

## 2. Clone the repository

```bash
git clone https://github.com/PatrickFu0416/titanic-survival-hw3.git
cd titanic-survival-hw3
mkdir -p src/data
```

Alternatively, in GitHub Desktop choose **File -> Clone Repository -> URL**, paste
the repository URL, choose a local path, and click Clone. Open the repository in
your text editor. All commands below run from this repository's root directory.

## 3. Download and place the data

1. Open the [Kaggle Titanic Data page](https://www.kaggle.com/competitions/titanic/data).
2. Sign in. If Kaggle prompts you to join the competition, review its rules and
   accept them yourself before downloading.
3. Download **train.csv** and **test.csv**, or use Download All and extract the ZIP.
4. Place the two CSV files directly in `src/data/` with these exact lowercase names:

```text
src/data/train.csv
src/data/test.csv
```

For example, if you downloaded `titanic.zip` to Downloads, on macOS/Linux or Git Bash:

```bash
unzip "$HOME/Downloads/titanic.zip" -d src/data
```

If your download saved the individual CSVs instead:

```bash
cp "$HOME/Downloads/train.csv" src/data/train.csv
cp "$HOME/Downloads/test.csv" src/data/test.csv
```

The competition files have 891 labeled training rows and 418 unlabeled test rows.
The training header includes `Survived`; the test header does not. Keep the provided
headers. Do not substitute a different Titanic dataset with a different schema.
The extra `gender_submission.csv` in the ZIP is an example submission; it is not
needed by either pipeline and is not a file of true outcomes.

`src/data/` is created locally because empty directories are not stored by Git.
`.gitignore` excludes the data directory, CSVs, output files, virtual environments,
credentials, `.DS_Store`, and editor files. `.dockerignore` also excludes data from
the build context. The Dockerfiles copy only code and dependency files into images.

## 4. Run from the terminal

Run either language, or both:

```bash
bash run.sh python
bash run.sh r
# Or run both sequentially:
bash run.sh all
```

The helper checks that the input files exist and Docker is running, builds the
selected image, mounts the data read-only, and saves the terminal log and result
files under `outputs/python/` or `outputs/r/`. It runs containers with your user ID
so generated files are owned by you on Linux as well. Paths containing spaces are
quoted. The script stops with an error if a build or model run fails.

### Equivalent explicit Python commands

```bash
mkdir -p outputs/python
docker build -f src/python/Dockerfile -t titanic-hw3-python .
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$(pwd)/src/data,target=/app/data,readonly" \
  --mount "type=bind,source=$(pwd)/outputs/python,target=/app/output" \
  titanic-hw3-python
```

The final `.` in the build command is required: it supplies the repository root as
the build context, where `requirements.txt` is located.

### Equivalent explicit R commands

```bash
mkdir -p outputs/r
docker build -f src/r/Dockerfile -t titanic-hw3-r .
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$(pwd)/src/data,target=/app/data,readonly" \
  --mount "type=bind,source=$(pwd)/outputs/r,target=/app/output" \
  titanic-hw3-r
```

The Python image pins Python 3.12.9 and every package in `requirements.txt`.
The R image pins R 4.5.1. R uses the `stats` and `utils` packages bundled with that
runtime; `install_packages.R` verifies their versions during the build. No CRAN
package installation is needed for the base-R pipeline.

## 5. What the models do

Both pipelines:

1. Validate file paths, required columns, passenger IDs, numeric values, categories,
   and labels. Keep all passenger rows and leave the original CSVs unchanged.
2. Print missing-value counts and training-set survival summaries by sex/class.
3. Add `FamilySize = SibSp + Parch + 1`.
4. Use `Age`, `Fare`, `FamilySize`, `Sex`, `Pclass`, and `Embarked`. Passenger IDs are
   retained for output matching, while `Name`, `Ticket`, and `Cabin` are omitted
   because this small baseline does not model high-cardinality text or sparse cabins.
5. Fit numeric medians and categorical modes only on the fitting subset. Apply the
   learned values to the holdout without refitting. Treat passenger class as categorical.
6. Fit logistic regression and use a 0.5 probability threshold. Python uses scaled
   numeric predictors, one-hot encoding and an L2-regularized scikit-learn model.
   R uses categorical factors and an unpenalized binomial `glm`.
7. Report an 80/20, fixed-seed, stratified holdout accuracy and majority-class baseline.
   Python and R use their own seed-42 splitting algorithms, so their holdout rows can
   differ. The scores are demonstrations, not a controlled comparison of languages.
8. Refit preprocessing and the model on all labeled training rows, report training
   accuracy, and predict the separate 418-row competition test set.

Training accuracy is a resubstitution score and usually optimistic. The holdout
score is the estimate on unseen labeled records. Neither is Kaggle leaderboard
accuracy. This baseline does not account for dependence among relatives traveling
together, so a row-level validation split can still be optimistic.

### Q18: Why competition-test accuracy is unavailable locally

[Kaggle explicitly withholds the test-set ground truth](https://www.kaggle.com/competitions/titanic/data).
Neither script can honestly calculate accuracy on `test.csv` alone. Both print
**NOT AVAILABLE**, leave that metric blank, and report a separate labeled holdout
score. `gender_submission.csv` contains baseline predictions, not actual survival
outcomes. Comparing predictions with it would measure agreement with a baseline,
not accuracy. No labels are inferred from another full-passenger dataset.

If an instructor separately provides **genuine** test outcomes, save a CSV with
exactly `PassengerId,Survived`, covering each test passenger exactly once. Both
scripts support `--test-labels` and join the labels by ID, not row order:

```bash
# Replace r with python for the Python container.
# Put the genuine instructor-supplied file in the ignored src/data directory.
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$(pwd)/src/data,target=/app/data,readonly" \
  --mount "type=bind,source=$(pwd)/outputs/r,target=/app/output" \
  titanic-hw3-r --data-dir /app/data --output-dir /app/output \
  --test-labels /app/data/genuine_test_labels.csv
```

Both scripts validate label coverage and values; a file's authenticity remains the
responsibility of its provider. The competition's example submission is never loaded
by the default workflow.

## 6. Output files

Each language writes its own directory:

- `predictions.csv`: `PassengerId,Survived`, in the original test-file order.
- `probabilities.csv`: passenger ID and estimated survival probability.
- `metrics.csv`: training accuracy, holdout accuracy, holdout baseline, and blank
  competition-test accuracy when genuine outcomes were not supplied.
- `run.log`: terminal output captured when using `run.sh`.

The prediction CSV is in Kaggle's required submission format. Results remain local
and ignored by Git. Container removal with `--rm` does not remove the mounted outputs.

## 7. Optional local runs

For Python, use Python 3.12 in a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Git Bash on Windows: source .venv/Scripts/activate
python -m pip install -r requirements.txt
python src/python/main.py
```

For a computer with R 4.5.1 installed:

```bash
Rscript src/r/install_packages.R
Rscript src/r/main.R
```

Docker is the primary supported route and avoids local runtime/version differences.

## 8. Troubleshooting

- **Missing train.csv/test.csv:** check spelling and place them directly under
  `src/data/`, not in an extra nested `titanic/` folder. Download the competition files.
- **Cannot connect to Docker:** start Docker Desktop and wait until its engine is
  running, then retry `docker info`. With Colima, start your Colima profile instead.
- **COPY requirements.txt failed:** run the build command from the repository root
  and keep its final `.`; do not build with `src/python` as the context.
- **Mount path does not exist:** run `mkdir -p src/data outputs/python outputs/r` and
  use the quoted mount commands above. Allow the project directory in Docker Desktop
  file sharing if your configuration requires it.
- **Windows path errors:** use Git Bash from the cloned directory. If Git Bash rewrites
  `/app` paths, prefix the `bash run.sh ...` command with `MSYS_NO_PATHCONV=1`.
- **Package/image download fails:** check the network and rerun the build. Cached
  layers are reused. No API key is required to build or run either model.
- **Invalid columns/values:** the scripts exit nonzero and name the problem. Preserve
  the original Kaggle headers and do not pass a prediction CSV as a training file.

## 9. Repository and Git workflow

```text
README.md
requirements.txt
run.sh
.gitignore
.dockerignore
src/
  python/Dockerfile
  python/main.py
  r/Dockerfile
  r/install_packages.R
  r/main.R
  data/                  # created locally; all contents ignored
outputs/                 # generated locally; all contents ignored
```

`main` is the release branch. `develop` collects reviewed work. Implementation
branches are created from `develop` and merged through pull requests with summaries;
`develop` is then promoted to `main`. See the repository's closed pull requests for
the setup and implementation history. The repo was cloned with GitHub Desktop and
opened in Visual Studio Code. `CODEOWNERS` is optional in the assignment; it is not
used here because code ownership requests alone do not restrict who can push.

## References

- [Kaggle Titanic data, schema and withheld labels](https://www.kaggle.com/competitions/titanic/data)
- [scikit-learn preprocessing pipelines](https://scikit-learn.org/1.5/modules/compose.html)
- [R generalized linear models](https://stat.ethz.ch/R-manual/R-release/library/stats/html/glm.html)
- [Docker bind mounts](https://docs.docker.com/engine/storage/bind-mounts/)
