locals {
  services = toset([
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    # needed to manage the service account and its project roles below
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com",
  ])
}

resource "google_project_service" "services" {
  for_each = local.services
  service  = each.value

  # On destroy, only forget the API instead of disabling it for the whole project.
  deletion_policy = "ABANDON"
}

# The offline store's dataset. The tables in it stay with the pipeline code (winterload.storage,
# created by storage.setup()), since their schemas change together with the backfill pipelines.
resource "google_bigquery_dataset" "offline_store" {
  dataset_id = var.dataset_id
  location   = var.region

  # Refuse to destroy the dataset while it still holds feature tables.
  delete_contents_on_destroy = false

  depends_on = [google_project_service.services]
}

# The identity the pipelines run as where there is no gcloud login, e.g. the Docker container.
# Its key is created with gcloud, not here: Terraform would keep the private key in its state.
resource "google_service_account" "pipeline" {
  account_id   = "winterload-pipeline"
  display_name = "Winterload pipeline"

  depends_on = [google_project_service.services]
}

resource "google_project_iam_member" "pipeline" {
  for_each = toset([
    "roles/bigquery.dataEditor", # read and write the feature tables
    "roles/bigquery.jobUser",    # run load jobs, queries and Feast's point-in-time joins
  ])
  project = var.project_id
  role    = each.value
  member  = google_service_account.pipeline.member
}
