# Run evidence

This folder holds the evidence that the scheduled flows ran: what Kestra started, what each run logged, what data it
left behind, and which scheduled runs are missing. Kestra is the scheduler that runs the winterload flows
(`daily_forecast`, `daily_monitoring`, `weekly_training`). A flow is one scheduled job, and an execution is one run of
it. The files here are generated. Do not edit them by hand.

## Export the evidence from the production Kestra

You need a browser and a login for https://atc.darker.dev. The Kestra API returns only the last 7 days unless the
`startDate` filter is set, so every URL below carries `startDate=2026-11-06`, the first day of the schedule. Save each
page with Cmd+S as a raw JSON file.

1. Sign in at https://atc.darker.dev in the browser.
2. Open this URL and save the page as `evidence/inbox/executions.json`:

   ```
   https://atc.darker.dev/api/v1/main/executions/search?filters[namespace][PREFIX]=winterload&filters[startDate][GREATER_THAN_OR_EQUAL_TO]=2026-11-06T00:00:00Z&size=1000
   ```

3. Open this URL and save the page as `evidence/inbox/logs_1.json`:

   ```
   https://atc.darker.dev/api/v1/main/logs/search?filters[namespace][PREFIX]=winterload&filters[startDate][GREATER_THAN_OR_EQUAL_TO]=2026-11-06T00:00:00Z&filters[level][GREATER_THAN_OR_EQUAL_TO]=INFO&size=1000&page=1
   ```

   The logs endpoint rejects `size` above 1000, so logs come in pages. Change `page=1` to `page=2` and save as
   `logs_2.json`, then continue with `page=3` and so on until a page shows an empty `results` list. Keep that
   empty page out of the folder or keep it, either works.
4. Run `make evidence`. It reads `evidence/inbox/` (override with `make evidence FROM=some/folder`) and rewrites the
   files in this folder.
5. Review `git diff docs/runs` and commit the result.

### Export from the local Kestra

`make evidence-local` skips the browser. It calls the local Kestra API itself, with `KESTRA_UI_USER` and
`KESTRA_UI_PASSWORD` from `.env`, sets the `startDate` filter and pages through the logs.

### When to export

Export every Monday after `weekly_training` has finished, and once more before the MS3 deadline.

## Files in this folder

| File | Content |
| --- | --- |
| `executions.csv` | One row per execution: `flow`, `execution_id`, `trigger` (`daily`, `weekly` or `manual`), `scheduled_date`, `started_at`, `ended_at`, `state`, `duration_s`. `scheduled_date` is the Zurich date of the start for triggered runs and stays empty for manual runs. |
| `missing_runs.csv` | One row per scheduled run that should exist and has no execution. Columns: `flow`, `scheduled_date`, `note`. |
| `logs/<flow>/<date>_<executionId>.log` | The log of one execution, sorted by time, one line per entry: `<timestamp> <LEVEL> <taskId> <message>`. `<date>` is the scheduled date, or the start date for manual runs. |
| `outputs.csv` | One row per Zurich day from the first day to yesterday: `energy_load_hours`, `weather_forecast_rows`, the hours stored per forecast role (`forecasts_<role>`) and `complete`. `complete` is true when energy has 24 hours and weather has 4 forecast rows per local hour. |
| `training_runs.csv` | MLflow runs of the training and monitoring experiments: `run_id`, `run_name`, `started_at`, `status`, `promoted`. |

## Reading `missing_runs.csv`

The file lists what the schedule promised and Kestra did not run. The expected runs are `daily_forecast` and
`daily_monitoring` on every Zurich day since the first day, and `weekly_training` on every Monday.

- A `daily_forecast` or `weekly_training` row is a real gap. Check the Kestra UI and the log folder for the cause.
- A `daily_monitoring` row with the note `no catch-up by design (recoverMissedSchedules: NONE)` is an intended skip. If
  Kestra was down at the scheduled time, it does not run the monitoring flow later. This is not a failure.
- An empty file means every expected run happened.
