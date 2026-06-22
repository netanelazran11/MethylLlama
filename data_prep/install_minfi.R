options(repos = c(CRAN = "https://cloud.r-project.org"))
.libPaths(c("~/R/libs", .libPaths()))

if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", lib = "~/R/libs")
}

BiocManager::install("minfi", lib = "~/R/libs", ask = FALSE, update = FALSE)
cat("Done\n")
