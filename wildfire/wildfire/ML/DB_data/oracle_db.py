import cx_Oracle
import numpy as np
import pandas as pd

# Oracle Instant Client 경로 설정
cx_Oracle.init_oracle_client(lib_dir="/Users/mmymacymac/Developer/Tools/instantclient_19_25")


class OracleDB:
    def __init__(self):
        self.conn = None
        self.cursor = None
        try:
            dsn = cx_Oracle.makedsn("localhost", 1521, service_name="XE")
            self.conn = cx_Oracle.connect(user="wildfire", password="1234", dsn=dsn)
            self.cursor = self.conn.cursor()
        except cx_Oracle.DatabaseError as e:
            print(f"❌ Oracle DB 연결 실패: {e}")
            self.conn = None
            self.cursor = None

    def get_features_by_region(self, region_name):
        """REGION_NAME으로 3개 테이블에서 모든 피처를 조회 후 통합"""
        if not self.conn:
            print("DB not connected.")
            return None

        # 3개 테이블을 JOIN하여 모든 컬럼을 조회
        sql_query = """
            SELECT 
                m.*,
                a1.*,
                a2.*
            FROM WILDFIRE_MAIN_FEATURES m
            LEFT JOIN WILDFIRE_AREA_FEATURES_1 a1 ON m.REGION_NAME = a1.REGION_NAME
            LEFT JOIN WILDFIRE_AREA_FEATURES_2 a2 ON m.REGION_NAME = a2.REGION_NAME
            WHERE m.REGION_NAME = :region_name
        """

        try:
            self.cursor.execute(sql_query, region_name=region_name)
            row = self.cursor.fetchone()
            if row:
                # DB 컬럼명(대문자)을 Python에서 사용할 소문자 키로 변환
                columns = [desc[0].lower() for desc in self.cursor.description]
                feature_dict = dict(zip(columns, row))

                # 중복 REGION_NAME 제거 (JOIN으로 인해 3번 나타남)
                cleaned_dict = {}
                for key, value in feature_dict.items():
                    if key == 'region_name' and 'region_name' in cleaned_dict:
                        continue  # 첫 번째 것만 유지
                    cleaned_dict[key] = value

                return cleaned_dict
            else:
                print(f"Warning: DB에서 {region_name}에 해당하는 데이터를 찾을 수 없습니다.")
                return None
        except Exception as e:
            print(f"Error fetching features for {region_name}: {e}")
            return None

    def insert_mountain_features(self, features):
        """모든 피처를 3개 테이블에 분산 저장"""
        if not self.conn:
            return

        region_name = features.get("REGION_NAME") or features.get("region_name")
        if not region_name:
            print("❌ REGION_NAME이 필요합니다.")
            return

        try:
            # features를 대문자로 변환
            features_upper = {k.upper(): v for k, v in features.items()}

            # 1. 각 테이블의 컬럼 정보 가져오기
            tables_info = {
                'WILDFIRE_MAIN_FEATURES': self._get_table_columns('WILDFIRE_MAIN_FEATURES'),
                'WILDFIRE_AREA_FEATURES_1': self._get_table_columns('WILDFIRE_AREA_FEATURES_1'),
                'WILDFIRE_AREA_FEATURES_2': self._get_table_columns('WILDFIRE_AREA_FEATURES_2')
            }

            # 2. 기존 데이터 삭제 (자식 테이블부터 삭제)
            delete_order = [
                'WILDFIRE_AREA_FEATURES_2',
                'WILDFIRE_AREA_FEATURES_1',
                'WILDFIRE_MAIN_FEATURES'
            ]
            for table_name in delete_order:
                delete_sql = f"DELETE FROM {table_name} WHERE REGION_NAME = :region_name"
                self.cursor.execute(delete_sql, {"region_name": region_name})

            # 3. 각 테이블별로 데이터 삽입
            success_count = 0

            # MAIN_FEATURES 테이블 삽입
            main_data = self._prepare_table_data(features_upper, tables_info['WILDFIRE_MAIN_FEATURES'])
            if self._insert_table_data('WILDFIRE_MAIN_FEATURES', main_data, region_name):
                success_count += 1

            # AREA_FEATURES_1 테이블 삽입
            area1_data = self._prepare_table_data(features_upper, tables_info['WILDFIRE_AREA_FEATURES_1'])
            if self._insert_table_data('WILDFIRE_AREA_FEATURES_1', area1_data, region_name):
                success_count += 1

            # AREA_FEATURES_2 테이블 삽입
            area2_data = self._prepare_table_data(features_upper, tables_info['WILDFIRE_AREA_FEATURES_2'])
            if self._insert_table_data('WILDFIRE_AREA_FEATURES_2', area2_data, region_name):
                success_count += 1

            self.conn.commit()
            print(f"✅ {region_name} 데이터가 성공적으로 {success_count}/3개 테이블에 저장되었습니다.")

        except cx_Oracle.DatabaseError as e:
            self.conn.rollback()
            print(f"❌ {region_name} 데이터 저장 오류: {e}")

    def _get_table_columns(self, table_name):
        """특정 테이블의 컬럼 목록 반환"""
        sql = """
            SELECT column_name, data_precision, data_scale
            FROM all_tab_columns
            WHERE table_name = :table_name AND column_name != 'UPDATED_AT'
        """
        self.cursor.execute(sql, {"table_name": table_name})
        return {row[0]: (row[1], row[2]) for row in self.cursor.fetchall()}

    def _prepare_table_data(self, features_upper, table_columns_info):
        """특정 테이블에 삽입할 데이터 준비"""
        table_data = {}

        for col_name, (precision, scale) in table_columns_info.items():
            if col_name in features_upper:
                value = features_upper[col_name]

                # 데이터 타입 변환 및 정밀도/스케일 맞추기
                if isinstance(value, (np.float32, np.float64)):
                    if scale is not None and scale >= 0:  # If scale is defined for NUMBER type
                        table_data[col_name] = round(float(value), scale)
                    else:  # For integer-like numbers or if scale is not applicable
                        table_data[col_name] = float(value)
                elif isinstance(value, (np.bool_, bool)):
                    table_data[col_name] = int(value)
                elif pd.isna(value):
                    table_data[col_name] = None
                else:
                    table_data[col_name] = value
            else:
                # 컬럼이 있지만 데이터가 없으면 None
                table_data[col_name] = None

        return table_data

    def _insert_table_data(self, table_name, data, region_name):
        """특정 테이블에 데이터 삽입"""
        if not data:
            return False

        columns = list(data.keys())
        placeholders = [f":{col}" for col in columns]

        insert_sql = f"""
            INSERT INTO {table_name} ({', '.join(columns)}) 
            VALUES ({', '.join(placeholders)})
        """

        try:
            self.cursor.execute(insert_sql, data)
            return True
        except cx_Oracle.DatabaseError as e:
            print(f"⚠️ {table_name} 삽입 실패 ({region_name}): {e}")
            return False

    def update_prectotcorr_0h(self, region_name, prectotcorr_value):
        """
        Updates only the PRECTOTCORR_0H column for a given region in WILDFIRE_AREA_FEATURES_2.
        """
        if not self.conn:
            print("DB 연결이 없습니다.")
            return False

        try:
            sql = """
                UPDATE WILDFIRE_AREA_FEATURES_2
                SET PRECTOTCORR_0H = :prectotcorr_value
                WHERE REGION_NAME = :region_name
            """
            self.cursor.execute(sql, {
                "prectotcorr_value": prectotcorr_value,
                "region_name": region_name
            })
            self.conn.commit()
            print(f"✅ {region_name}의 PRECTOTCORR_0H 업데이트 성공.")
            return True
        except cx_Oracle.DatabaseError as e:
            self.conn.rollback()
            print(f"❌ {region_name}의 PRECTOTCORR_0H 업데이트 실패: {e}")
            return False

    def close(self):
        if self.cursor: self.cursor.close()
        if self.conn: self.conn.close()
