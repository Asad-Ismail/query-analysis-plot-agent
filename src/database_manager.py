"""Database connection and permission management"""
import sqlite3
import yaml
from typing import List, Dict, Optional

class DatabaseManager:
    """Manages multiple database connections"""
    
    def __init__(self, config_path: str = "config/databases.yaml"):
        self.databases = {}
        self.load_config(config_path)
        self.schema_cache = {}
    
    def load_config(self, config_path: str):
        """Load database configurations"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for name, settings in config['databases'].items():
            self.databases[name] = {
                'path': settings['path'],
                'type': settings.get('type', 'sqlite')
            }
    
    def get_connection(self, db_name: str):
        """Get database connection"""
        if db_name not in self.databases:
            raise ValueError(f"Database {db_name} not found")
        
        db_info = self.databases[db_name]
        return sqlite3.connect(db_info['path'])
    
    def get_schema(self, db_name: str) -> Dict[str, List[str]]:
        """Get database schema"""

        if db_name in self.schema_cache:
            return self.schema_cache[db_name]
        conn = self.get_connection(db_name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns_info = cursor.fetchall()
            columns = [{'name': col[1], 'type': col[2]} for col in columns_info]
            '''
            cursor.execute(f"PRAGMA foreign_key_list({table})")
            fk_info = cursor.fetchall()
            foreign_keys = [
                {
                    'from': fk[3],
                    'to_table': fk[2],
                    'to_column': fk[4]
                } for fk in fk_info
            ]
            '''
            schema[table] = {'columns': columns}
            #'foreign_keys': foreign_keys}
        
        conn.close()
        self.schema_cache[db_name] = schema
        return schema

class PermissionManager:
    """Manages user permissions"""
    
    def __init__(self, config_path: str = "config/permissions.yaml"):
        self.permissions = {}
        self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load permission configurations"""
        with open(config_path, 'r') as f:
            self.permissions = yaml.safe_load(f)
    
    def get_allowed_tables(self, user_role: str, db_name: str) -> Optional[List[str]]:
        """Get list of tables user can access"""
        role_perms = self.permissions.get('roles', {}).get(user_role, {})
        db_perms = role_perms.get(db_name, {})
        return db_perms.get('tables')
    
    def can_access(self, user_role: str, db_name: str, table: str) -> bool:
        """Check if user can access specific table"""
        allowed = self.get_allowed_tables(user_role, db_name)
        if allowed is None:  # None means all tables
            return True
        return table in allowed
