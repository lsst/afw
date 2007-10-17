# Schema for per visit MovingObjectPrediction tables.
CREATE TABLE IF NOT EXISTS MovingObjectPredictionTemplate
(
    orbit_id BIGINT NOT NULL,
    ra_deg   DOUBLE NOT NULL,
    dec_deg  DOUBLE NOT NULL,
    mjd      DOUBLE NOT NULL,
    smia     DOUBLE NOT NULL,
    smaa     DOUBLE NOT NULL,
    pa       DOUBLE NOT NULL,
    mag      DOUBLE NOT NULL
) ENGINE=MyISAM;


