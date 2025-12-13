DROP TABLE IF EXISTS permits;
DROP TABLE IF EXISTS properties;
DROP TABLE IF EXISTS neighborhood_profiles;

CREATE TABLE neighborhood_profiles (
  neighborhood TEXT PRIMARY KEY,
  population INTEGER,
  median_income REAL,
  unemployment_rate REAL,
  percent_immigrant REAL,
  percent_seniors REAL,
  last_updated TEXT
);

CREATE TABLE properties (
  property_id INTEGER PRIMARY KEY AUTOINCREMENT,
  account_number INTEGER,
  house_number TEXT,
  street_name TEXT,
  neighborhood_id INTEGER,
  neighborhood TEXT,
  assessed_value REAL,
  tax_class TEXT,
  garage TEXT,
  ward TEXT,
  latitude REAL,
  longitude REAL,
  point_location TEXT
);

CREATE INDEX IF NOT EXISTS idx_properties_neighborhood ON properties(neighborhood);
CREATE INDEX IF NOT EXISTS idx_properties_account ON properties(account_number);

CREATE TABLE permits (
  permit_id INTEGER PRIMARY KEY AUTOINCREMENT,
  issued_date TEXT,
  year INTEGER,
  month_number INTEGER,
  job_category TEXT,
  work_type TEXT,
  project_value REAL,
  raw_address TEXT,
  neighborhood TEXT,
  latitude REAL,
  longitude REAL,
  FOREIGN KEY (neighborhood) REFERENCES neighborhood_profiles(neighborhood)
);
