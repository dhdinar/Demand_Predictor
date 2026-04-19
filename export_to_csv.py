from DemandPrediction.api_service import export_mysql_to_csv

# MySQL connection info (from your Django settings)
host = "localhost"
port = 3306
dbname = "tumii"
user = "root"
password = ""

# Path to your SQL file
with open("sql/weekly_demand_features.sql", "r", encoding="utf-8") as f:
    sql_query = f.read()

# Export to CSV
export_mysql_to_csv(
    host=host,
    port=port,
    dbname=dbname,
    user=user,
    password=password,
    output_csv="data.csv",
    sql_query=sql_query,
)

print("Exported data to data.csv")
