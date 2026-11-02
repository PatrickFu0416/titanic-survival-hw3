# R's logistic regression and CSV functions are bundled with the pinned runtime.
# Verify them at image-build time; this pipeline needs no third-party CRAN packages.
required_versions <- c(stats = "4.5.1", utils = "4.5.1")
for (package in names(required_versions)) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(sprintf("Required bundled package is missing: %s", package))
  }
  actual <- as.character(utils::packageVersion(package))
  if (actual != required_versions[[package]]) {
    stop(sprintf("Expected %s %s; found %s. Use the pinned R 4.5.1 image.",
                 package, required_versions[[package]], actual))
  }
  cat(sprintf("Verified bundled dependency: %s == %s\n", package, actual))
}
cat("Package setup complete. No external packages need installation.\n")
