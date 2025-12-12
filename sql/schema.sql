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
  assessment_year INTEGER,
  assessed_value REAL,
  neighborhood TEXT,
  house_number TEXT,
  street_name TEXT,
  street_type TEXT,
  postal_code TEXT,
  property_type TEXT,
  year_built INTEGER,
  lot_size REAL,
  latitude REAL,
  longitude REAL,
  raw_address TEXT,
  FOREIGN KEY (neighborhood) REFERENCES neighborhood_profiles(neighborhood)
);

CREATE TABLE permits (
  permit_id INTEGER PRIMARY KEY AUTOINCREMENT,
  issued_date TEXT,
  permit_type TEXT,
  work_class_group TEXT,
  project_value REAL,
  contractor TEXT,
  neighborhood TEXT,
  latitude REAL,
  longitude REAL,
  raw_address TEXT,
  FOREIGN KEY (neighborhood) REFERENCES neighborhood_profiles(neighborhood)
);
