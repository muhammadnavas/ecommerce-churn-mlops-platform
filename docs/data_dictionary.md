# Data Dictionary & Schema Documentation

## 1. Data Quality Validation Rules

| Rule ID | Check Name | Condition | Action on Failure | Log Code |
|:---|:---|:---|:---|:---|
| **DQ-01** | Missing Customer ID | `CustomerID is NULL or empty` | Reject row | `MISSING_CUSTOMER_ID` |
| **DQ-02** | Invalid Price | `UnitPrice <= 0` | Reject row | `INVALID_UNIT_PRICE` |
| **DQ-03** | Zero Quantity | `Quantity == 0` | Reject row | `ZERO_QUANTITY` |
| **DQ-04** | Date Format | `InvoiceDate != %Y-%m-%d %H:%M:%S` | Reject row | `CORRUPT_DATE_FORMAT` |
| **DQ-05** | Unsolicited Cancellation | `Quantity < 0 AND InvoiceNo !startswith 'C'` | Reject row | `NEGATIVE_QUANTITY_UNSOLICITED` |

---

## 2. Data Warehouse Star Schema

### Dimension: `dim_customer`
| Column Name | Data Type | Key Type | Description |
|:---|:---|:---|:---|
| `customer_id` | TEXT | Primary Key | Unique 5-digit customer identifier |
| `primary_country` | TEXT | - | Primary country associated with customer |
| `first_purchase_date` | TEXT | - | Date of customer's first transaction |
| `last_purchase_date` | TEXT | - | Date of customer's latest transaction |
| `total_orders` | INTEGER | - | Total completed order count |

### Dimension: `dim_product`
| Column Name | Data Type | Key Type | Description |
|:---|:---|:---|:---|
| `stock_code` | TEXT | Primary Key | Product stock code |
| `description` | TEXT | - | Item description |
| `avg_unit_price` | REAL | - | Average unit price (£) |

### Dimension: `dim_geography`
| Column Name | Data Type | Key Type | Description |
|:---|:---|:---|:---|
| `country_id` | INTEGER | Primary Key | Auto-increment country key |
| `country_name` | TEXT | Unique | Country name |

### Dimension: `dim_date`
| Column Name | Data Type | Key Type | Description |
|:---|:---|:---|:---|
| `date_key` | INTEGER | Primary Key | Integer date key (YYYYMMDD) |
| `full_date` | TEXT | Unique | Full date (YYYY-MM-DD) |
| `year` | INTEGER | - | Calendar year |
| `quarter` | INTEGER | - | Calendar quarter (1-4) |
| `month` | INTEGER | - | Calendar month (1-12) |
| `day` | INTEGER | - | Day of month |
| `day_of_week` | INTEGER | - | Day of week (0=Mon, 6=Sun) |
| `is_weekend` | INTEGER | - | Indicator (1=Weekend, 0=Weekday) |

### Fact Table: `fact_sales`
| Column Name | Data Type | Key Type | Description |
|:---|:---|:---|:---|
| `sales_id` | INTEGER | Primary Key | Surrogate primary key |
| `invoice_no` | TEXT | - | Transaction invoice number |
| `customer_id` | TEXT | Foreign Key | References `dim_customer.customer_id` |
| `stock_code` | TEXT | Foreign Key | References `dim_product.stock_code` |
| `country_id` | INTEGER | Foreign Key | References `dim_geography.country_id` |
| `date_key` | INTEGER | Foreign Key | References `dim_date.date_key` |
| `invoice_date` | TEXT | - | Exact transaction timestamp |
| `quantity` | INTEGER | - | Number of items purchased |
| `unit_price` | REAL | - | Unit price (£) |
| `total_amount` | REAL | - | Line total (£) = quantity * unit_price |
| `is_cancelled` | INTEGER | - | Indicator (1=Cancelled 'C' order, 0=Valid) |

### Analytical Data Mart: `fact_customer_rfm`
| Column Name | Data Type | Key Type | Description |
|:---|:---|:---|:---|
| `customer_id` | TEXT | Primary Key | References `dim_customer.customer_id` |
| `snapshot_date` | TEXT | - | Analysis snapshot date |
| `recency_days` | INTEGER | - | Days elapsed since last order |
| `frequency` | INTEGER | - | Total completed order count |
| `monetary` | REAL | - | Total revenue generated (£) |
| `avg_order_value` | REAL | - | Average revenue per order (£) |
| `total_items` | INTEGER | - | Cumulative items purchased |
| `cancellation_rate` | REAL | - | Ratio of cancelled orders |
| `churn_label` | INTEGER | Target | Binary indicator (1=Churned >90d inactive, 0=Active) |
