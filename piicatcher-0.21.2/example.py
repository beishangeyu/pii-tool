from dbcat.api import open_catalog, add_postgresql_source, add_sqlite_source
from piicatcher.api import scan_database, ScanTypeEnum, OutputFormat


catalog = open_catalog(app_dir='/tmp/.config/piicatcher', path=':memory:', secret='my_secret')

with catalog.managed_session:
    # Add a postgresql source
    # source = add_postgresql_source(catalog=catalog, name="piidata", uri="127.0.0.1", username="eyu",
    #                                 password="eyu", database="pii_data")
    source = add_sqlite_source(catalog=catalog, name="piidata", path="./sample_data.db")
    
    # 基本扫描（默认参数，元数据扫描）
    print("=== 基本扫描（元数据） ===")
    output = scan_database(
        catalog=catalog, 
        source=source,
        incremental=False  # 禁用增量扫描
    )
    print(output)
    
    # 扫描实际数据内容而非仅元数据
    print("\n=== 数据内容扫描 ===")
    output_data = scan_database(
        catalog=catalog, 
        source=source,
        scan_type=ScanTypeEnum.data,  # 扫描实际数据
        incremental=False  # 禁用增量扫描
    )
    print(output_data)
    
    # 返回JSON格式的详细结果
    print("\n=== JSON格式输出 ===")
    output_json = scan_database(
        catalog=catalog, 
        source=source,
        output_format=OutputFormat.json,  # JSON格式输出
        incremental=False  # 禁用增量扫描
    )
    print(output_json)
    
    # 显示所有列（包括非PII列）
    print("\n=== 显示所有列 ===")
    output_all = scan_database(
        catalog=catalog, 
        source=source,
        list_all=True,  # 显示所有列
        incremental=False  # 禁用增量扫描
    )
    print(output_all)
    
    # 过滤特定表
    print("\n=== 过滤特定表 ===")
    output_filtered = scan_database(
        catalog=catalog, 
        source=source,
        include_table_regex=["sample.*"],  # 修改正则表达式匹配
        scan_type=ScanTypeEnum.data,
        output_format=OutputFormat.json,
        incremental=False  # 禁用增量扫描
    )
    print(output_filtered)