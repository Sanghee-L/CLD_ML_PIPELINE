CREATE TABLE product (
    product_id TEXT PRIMARY KEY,
    modality TEXT,
    target_indication TEXT
);

CREATE TABLE host_cell (
    host_id TEXT PRIMARY KEY,
    species TEXT,
    genetic_background TEXT
);

CREATE TABLE vector (
    vector_id TEXT PRIMARY KEY,
    promoter TEXT,
    signal_peptide TEXT,
    backbone TEXT,
    copy_number_est INTEGER
);

CREATE TABLE cell_line (
    cell_line_id TEXT PRIMARY KEY,
    product_id TEXT,
    host_id TEXT,
    vector_id TEXT,
    transfection_date DATE,
    transfection_method TEXT,
    selection_marker TEXT,
    FOREIGN KEY(product_id) REFERENCES product(product_id),
    FOREIGN KEY(host_id) REFERENCES host_cell(host_id),
    FOREIGN KEY(vector_id) REFERENCES vector(vector_id)
);

CREATE TABLE clone (
    clone_id TEXT PRIMARY KEY,
    cell_line_id TEXT,
    isolation_method TEXT,
    clone_rank INTEGER,
    FOREIGN KEY(cell_line_id) REFERENCES cell_line(cell_line_id)
);

CREATE TABLE passage (
    passage_id TEXT PRIMARY KEY,
    clone_id TEXT,
    passage_number INTEGER,
    culture_duration INTEGER,
    phase TEXT,
    FOREIGN KEY(clone_id) REFERENCES clone(clone_id)
);

CREATE TABLE batch (
    batch_id TEXT PRIMARY KEY,
    experiment_type TEXT,
    run_date DATE,
    platform TEXT,
    operator TEXT
);

CREATE TABLE assay_result (
    assay_id TEXT PRIMARY KEY,
    passage_id TEXT,
    batch_id TEXT,
    assay_type TEXT,
    value REAL,
    unit TEXT,
    method TEXT,
    FOREIGN KEY(passage_id) REFERENCES passage(passage_id),
    FOREIGN KEY(batch_id) REFERENCES batch(batch_id)
);

CREATE TABLE omics_sample (
    omics_id TEXT PRIMARY KEY,
    clone_id TEXT,
    passage_id TEXT,
    batch_id TEXT,
    omics_type TEXT,
    timepoint TEXT,
    FOREIGN KEY(clone_id) REFERENCES clone(clone_id),
    FOREIGN KEY(passage_id) REFERENCES passage(passage_id),
    FOREIGN KEY(batch_id) REFERENCES batch(batch_id)
);

CREATE TABLE process_condition (
    condition_id TEXT PRIMARY KEY,
    passage_id TEXT,
    culture_mode TEXT,
    temp REAL,
    pH REAL,
    feed_strategy TEXT,
    medium TEXT,
    FOREIGN KEY(passage_id) REFERENCES passage(passage_id)
);

CREATE TABLE stability_test (
    stability_id TEXT PRIMARY KEY,
    clone_id TEXT,
    start_passage INTEGER,
    end_passage INTEGER,
    productivity_drop_pct REAL,
    metric_type TEXT,
    evaluation_method TEXT,
    FOREIGN KEY(clone_id) REFERENCES clone(clone_id)
);