DROP VIEW IF EXISTS v_permits_monthly;
DROP VIEW IF EXISTS v_properties_by_neighborhood;
DROP VIEW IF EXISTS v_neighborhood_monthly;

CREATE VIEW v_permits_monthly AS
SELECT
  neighborhood,
  year,
  month_number,
  COUNT(*) AS permits_count,
  SUM(COALESCE(project_value, 0)) AS total_construction_value,
  AVG(project_value) AS avg_construction_value
FROM permits
GROUP BY neighborhood, year, month_number;

CREATE VIEW v_properties_by_neighborhood AS
SELECT
  neighborhood,
  COUNT(*) AS n_properties,
  AVG(assessed_value) AS avg_assessed_value
FROM properties
GROUP BY neighborhood;

CREATE VIEW v_neighborhood_monthly AS
SELECT
  p.neighborhood,
  p.year,
  p.month_number,
  p.permits_count,
  p.total_construction_value,
  p.avg_construction_value,
  pr.n_properties,
  pr.avg_assessed_value
FROM v_permits_monthly p
LEFT JOIN v_properties_by_neighborhood pr
  ON p.neighborhood = pr.neighborhood;
