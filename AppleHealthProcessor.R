cran <- getOption("repos")
cran["CRAN"] <- "https://cloud.r-project.org"
options(repos = cran)
install.packages(c("xml2", "lubridate", "dplyr", "tidyr", "zip", "fs"))
library(xml2)
library(lubridate)
library(dplyr)
library(tidyr)
library(zip)
library(fs)

process_apple_health <- function(xml_path, output_path = getwd()) {
  # Ensure output path exists
  dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

  # Read XML
  message("Reading XML file...")
  xml_data <- read_xml(xml_path)

  # Extract all records
  records <- xml_find_all(xml_data, "//Record")

  # Convert to data frame
  message("Converting records to data frame...")
  health_data <- data.frame(
    type = xml_attr(records, "type"),
    start_date = ymd_hms(xml_attr(records, "startDate")),
    value = as.numeric(xml_attr(records, "value"))
  )

  # Get date range
  min_date <- min(health_data$start_date, na.rm = TRUE)
  max_date <- max(health_data$start_date, na.rm = TRUE)

  message(sprintf("Processing data from %s to %s",
                 format(min_date, "%Y-%m-%d"),
                 format(max_date, "%Y-%m-%d")))

  # Create year-month column
  health_data$year_month <- format(health_data$start_date, "%Y-%m")

  # Split and export by month
  csv_files <- character()
  # Rename columns
  colnames(health_data) <- c("RecordType", "StartDate", "Value", "year_month")
  # Group by year-month and write separate CSV files
  message("Exporting monthly CSV files...")
  for (ym in unique(health_data$year_month)) {
    month_data <- health_data[health_data$year_month == ym,]
    csv_filename <- file.path(output_path, paste0(ym, ".csv"))
    write.csv(month_data, csv_filename, row.names = FALSE)
    csv_files <- c(csv_files, csv_filename)
  }

  # Create zip file
  if (length(csv_files) > 0) {
    zip_filename <- file.path(output_path,
                            sprintf("apple_health_data_%s-%s.zip",
                                  year(min_date),
                                  year(max_date)))

    message("Creating zip file...")
    zip::zip(zip_filename,
            files = csv_files,
            mode = "cherry-pick")

    # Clean up CSV files
    message("Cleaning up CSV files...")
    file.remove(csv_files)

    message(sprintf("Data exported to %s", zip_filename))
    return(zip_filename)
  } else {
    message("No data to export")
    return(NULL)
  }
}

# Main execution
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) > 0) {
    xml_path <- args[1]
    output_path <- if (length(args) > 1) args[2] else getwd()
    process_apple_health(xml_path, output_path)
  } else {
    message("Please provide the path to the Apple Health Export XML file")
    message("Usage: Rscript AppleHealthProcessor.R <xml_path> [output_path]")
  }
}
