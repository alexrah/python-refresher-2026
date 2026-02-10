# Python Refresher: Pydantic, Pandas & NumPy
## For TypeScript Developers

---

## Table of Contents
1. [Python Basics Recap](#python-basics-recap)
2. [Pydantic - Data Validation](#pydantic)
3. [Pandas - Data Manipulation](#pandas)
4. [NumPy - Numerical Computing](#numpy)
5. [Putting It All Together](#integration)

---

## Python Basics Recap

### Type Hints (Similar to TypeScript)

```python
# Python with type hints
from typing import List, Dict, Optional, Union

def greet(name: str, age: int) -> str:
    return f"Hello {name}, you are {age} years old"

# TypeScript equivalent:
# function greet(name: string, age: number): string {
#     return `Hello ${name}, you are ${age} years old`;
# }

# Collections
numbers: List[int] = [1, 2, 3]
user_data: Dict[str, Union[str, int]] = {"name": "Alice", "age": 30}
optional_value: Optional[str] = None  # Like string | null in TS
```

### List Comprehensions (Think Array Methods)

```python
# Python list comprehension
squares = [x**2 for x in range(10)]

# TypeScript equivalent:
# const squares = Array.from({length: 10}, (_, x) => x**2);
# or: [...Array(10).keys()].map(x => x**2)

# With filtering
evens = [x for x in range(10) if x % 2 == 0]
# TS: Array.from({length: 10}, (_, x) => x).filter(x => x % 2 === 0)
```

### Key Differences from TypeScript

```python
# 1. No semicolons, indentation matters
if True:
    print("Indented block")

# 2. Different equality operators
x == y   # Value equality (like === in TS)
x is y   # Reference equality

# 3. None instead of null/undefined
value = None

# 4. Different boolean values
is_active = True  # Capital T
is_disabled = False  # Capital F

# 5. f-strings for interpolation
name = "Alice"
message = f"Hello {name}"  # Like `Hello ${name}` in TS
```

---

## Pydantic

Pydantic is like Zod or io-ts for TypeScript - runtime data validation with type safety.

### Installation

```bash
pip install pydantic --break-system-packages
```

### Basic Models

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

class User(BaseModel):
    id: int
    username: str
    email: str
    age: Optional[int] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Custom validation
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v.lower()
    
    @validator('age')
    def validate_age(cls, v):
        if v is not None and (v < 0 or v > 150):
            raise ValueError('Age must be between 0 and 150')
        return v

# Creating instances
user = User(
    id=1,
    username="alice",
    email="ALICE@EXAMPLE.COM"  # Will be lowercased
)

print(user.email)  # alice@example.com
print(user.dict())  # Convert to dictionary
print(user.json())  # Convert to JSON string
```

### TypeScript Comparison

```typescript
// In TypeScript with Zod:
// import { z } from 'zod';
// 
// const UserSchema = z.object({
//     id: z.number(),
//     username: z.string(),
//     email: z.string().email().transform(s => s.toLowerCase()),
//     age: z.number().min(0).max(150).optional(),
//     is_active: z.boolean().default(true),
//     created_at: z.date().default(() => new Date())
// });
// 
// type User = z.infer<typeof UserSchema>;
```

### Nested Models

```python
from pydantic import BaseModel
from typing import List

class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str

class Company(BaseModel):
    name: str
    employees: int

class Employee(BaseModel):
    id: int
    name: str
    email: str
    address: Address
    company: Company
    skills: List[str]

# Usage
employee = Employee(
    id=1,
    name="John Doe",
    email="john@example.com",
    address={
        "street": "123 Main St",
        "city": "New York",
        "country": "USA",
        "postal_code": "10001"
    },
    company={
        "name": "Tech Corp",
        "employees": 500
    },
    skills=["Python", "TypeScript", "SQL"]
)

print(employee.address.city)  # New York
```

### Config and Advanced Features

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    id: int
    name: str
    price: float = Field(..., gt=0, description="Price must be positive")
    quantity: int = Field(default=0, ge=0)
    tags: List[str] = []
    
    class Config:
        # Allow population by field name or alias
        populate_by_name = True
        # Validate on assignment
        validate_assignment = True
        # Generate JSON schema
        json_schema_extra = {
            "example": {
                "id": 1,
                "name": "Laptop",
                "price": 999.99,
                "quantity": 10,
                "tags": ["electronics", "computers"]
            }
        }

# Validation on assignment
product = Product(id=1, name="Phone", price=599.99)
# product.price = -10  # Would raise ValidationError
```

### Parsing and Serialization

```python
import json
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    value: float

# Parse from dictionary
data = {"name": "Widget", "value": 42.5}
item = Item(**data)

# Parse from JSON
json_str = '{"name": "Gadget", "value": 100.0}'
item = Item.model_validate_json(json_str)

# Serialize
print(item.model_dump())  # To dict
print(item.model_dump_json())  # To JSON string

# Partial updates
item_dict = item.model_dump(exclude={'value'})
```

---

## Pandas

Pandas is like a supercharged data manipulation library - think lodash + SQL for tabular data.

### Installation

```bash
pip install pandas --break-system-packages
```

### Creating DataFrames

```python
import pandas as pd
import numpy as np

# From dictionary (like creating from object in TS)
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 40],
    'city': ['NYC', 'SF', 'LA', 'NYC'],
    'salary': [70000, 80000, 90000, 95000]
}
df = pd.DataFrame(data)

print(df)
#       name  age city  salary
# 0    Alice   25  NYC   70000
# 1      Bob   30   SF   80000
# 2  Charlie   35   LA   90000
# 3    David   40  NYC   95000

# From list of dictionaries
records = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30}
]
df2 = pd.DataFrame(records)

# From CSV
# df = pd.read_csv('data.csv')
# df = pd.read_excel('data.xlsx')
# df = pd.read_json('data.json')
```

### Basic Operations

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 40],
    'city': ['NYC', 'SF', 'LA', 'NYC'],
    'salary': [70000, 80000, 90000, 95000]
})

# View data
print(df.head())     # First 5 rows
print(df.tail(2))    # Last 2 rows
print(df.info())     # Column types and null counts
print(df.describe()) # Statistical summary

# Column access (like object property access)
print(df['name'])         # Single column -> Series
print(df[['name', 'age']]) # Multiple columns -> DataFrame

# Row access
print(df.iloc[0])         # By index position (like array[0])
print(df.loc[0])          # By label

# Filtering (like Array.filter)
young = df[df['age'] < 35]
nyc_high_salary = df[(df['city'] == 'NYC') & (df['salary'] > 70000)]

# TypeScript equivalent of filtering:
# const young = data.filter(person => person.age < 35);
# const nycHighSalary = data.filter(p => p.city === 'NYC' && p.salary > 70000);
```

### Data Manipulation

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 40],
    'salary': [70000, 80000, 90000, 95000]
})

# Adding columns (like spread operator or Object.assign)
df['bonus'] = df['salary'] * 0.1
df['full_salary'] = df['salary'] + df['bonus']

# Map/Transform (like Array.map)
df['age_group'] = df['age'].apply(lambda x: 'Young' if x < 30 else 'Senior')

# TypeScript equivalent:
# data.map(p => ({ ...p, age_group: p.age < 30 ? 'Young' : 'Senior' }))

# Sort (like Array.sort)
sorted_df = df.sort_values('salary', ascending=False)

# GroupBy (like lodash groupBy + reduce)
grouped = df.groupby('age_group')['salary'].mean()
# TypeScript: _.groupBy + _.mapValues with reduce

# Aggregations
summary = df.groupby('age_group').agg({
    'salary': ['mean', 'sum', 'count'],
    'age': 'max'
})
```

### Data Cleaning

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'name': ['Alice', 'Bob', None, 'David'],
    'age': [25, 30, 35, np.nan],
    'salary': [70000, None, 90000, 95000]
})

# Handle missing values
print(df.isnull().sum())  # Count nulls per column

# Drop rows with any null
cleaned = df.dropna()

# Fill nulls
filled = df.fillna({
    'age': df['age'].mean(),
    'salary': 0,
    'name': 'Unknown'
})

# Drop duplicates
df_unique = df.drop_duplicates()

# Rename columns
df_renamed = df.rename(columns={
    'name': 'employee_name',
    'age': 'employee_age'
})

# Select specific columns with types
numeric_cols = df.select_dtypes(include=[np.number])
string_cols = df.select_dtypes(include=['object'])
```

### Merging and Joining

```python
import pandas as pd

# Like SQL JOIN or TypeScript Array merging
employees = pd.DataFrame({
    'emp_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'dept_id': [10, 20, 10]
})

departments = pd.DataFrame({
    'dept_id': [10, 20, 30],
    'dept_name': ['Engineering', 'Sales', 'HR']
})

# Inner join (intersection)
merged = pd.merge(employees, departments, on='dept_id', how='inner')

# Left join (all from left, matching from right)
merged = pd.merge(employees, departments, on='dept_id', how='left')

# Outer join (all from both)
merged = pd.merge(employees, departments, on='dept_id', how='outer')

# Concatenate (like Array.concat)
df1 = pd.DataFrame({'A': [1, 2]})
df2 = pd.DataFrame({'A': [3, 4]})
combined = pd.concat([df1, df2], ignore_index=True)
```

### Time Series Operations

```python
import pandas as pd
from datetime import datetime, timedelta

# Create date range
dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')

df = pd.DataFrame({
    'date': dates,
    'value': [100, 110, 105, 115, 120, 118, 125, 130, 128, 135]
})

# Set date as index
df.set_index('date', inplace=True)

# Resample (aggregate by time period)
weekly = df.resample('W').mean()

# Rolling window (moving average)
df['moving_avg'] = df['value'].rolling(window=3).mean()

# Shift (lag/lead)
df['previous_day'] = df['value'].shift(1)
df['next_day'] = df['value'].shift(-1)
```

---

## NumPy

NumPy provides efficient numerical operations on arrays - like typed arrays in JS but much more powerful.

### Installation

```bash
pip install numpy --break-system-packages
```

### Creating Arrays

```python
import numpy as np

# From lists
arr = np.array([1, 2, 3, 4, 5])
print(arr.dtype)  # int64

# 2D array (matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.shape)  # (2, 3)

# Special arrays
zeros = np.zeros((3, 4))        # 3x4 array of zeros
ones = np.ones((2, 3))          # 2x3 array of ones
identity = np.eye(3)            # 3x3 identity matrix
random_arr = np.random.rand(5)  # 5 random numbers [0, 1)
range_arr = np.arange(0, 10, 2) # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5) # [0, 0.25, 0.5, 0.75, 1]

# With specific dtype
float_arr = np.array([1, 2, 3], dtype=np.float64)
```

### Array Operations

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Element-wise operations (vectorized, much faster than loops)
print(arr + 10)      # [11, 12, 13, 14, 15]
print(arr * 2)       # [2, 4, 6, 8, 10]
print(arr ** 2)      # [1, 4, 9, 16, 25]
print(np.sqrt(arr))  # [1., 1.414, 1.732, 2., 2.236]

# Array-to-array operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(arr1 + arr2)   # [5, 7, 9]
print(arr1 * arr2)   # [4, 10, 18]

# Aggregations
print(arr.sum())     # 15
print(arr.mean())    # 3.0
print(arr.std())     # Standard deviation
print(arr.min())     # 1
print(arr.max())     # 5

# Boolean operations
print(arr > 3)       # [False, False, False, True, True]
filtered = arr[arr > 3]  # [4, 5]
```

### Indexing and Slicing

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# Indexing (like regular arrays)
print(arr[0])        # 10
print(arr[-1])       # 50 (last element)

# Slicing [start:end:step]
print(arr[1:4])      # [20, 30, 40]
print(arr[:3])       # [10, 20, 30]
print(arr[::2])      # [10, 30, 50] (every 2nd element)

# 2D arrays
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix[0, 1])  # 2 (row 0, col 1)
print(matrix[1])     # [4, 5, 6] (entire row 1)
print(matrix[:, 1])  # [2, 5, 8] (entire column 1)
print(matrix[0:2, 1:3])  # [[2, 3], [5, 6]]

# Boolean indexing
arr = np.array([1, 2, 3, 4, 5])
mask = arr > 2
print(arr[mask])     # [3, 4, 5]
```

### Reshaping and Combining

```python
import numpy as np

arr = np.arange(12)  # [0, 1, 2, ..., 11]

# Reshape
reshaped = arr.reshape(3, 4)  # 3x4 matrix
# [[0, 1, 2, 3],
#  [4, 5, 6, 7],
#  [8, 9, 10, 11]]

# Flatten
flattened = reshaped.flatten()  # Back to 1D

# Transpose
transposed = reshaped.T

# Stack arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

vstacked = np.vstack([arr1, arr2])  # Vertical stack
# [[1, 2, 3],
#  [4, 5, 6]]

hstacked = np.hstack([arr1, arr2])  # Horizontal stack
# [1, 2, 3, 4, 5, 6]

# Concatenate
concatenated = np.concatenate([arr1, arr2])
```

### Linear Algebra

```python
import numpy as np

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise multiplication
print(A * B)

# Matrix multiplication (dot product)
print(A @ B)         # or np.dot(A, B)
print(np.matmul(A, B))

# Inverse
inv_A = np.linalg.inv(A)

# Determinant
det = np.linalg.det(A)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Solve linear equations Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
```

### Statistical Operations

```python
import numpy as np

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Basic statistics
print(np.mean(data))      # Mean
print(np.median(data))    # Median
print(np.std(data))       # Standard deviation
print(np.var(data))       # Variance
print(np.percentile(data, 75))  # 75th percentile

# Random sampling
random_sample = np.random.choice(data, size=5, replace=False)

# Normal distribution
normal = np.random.normal(loc=0, scale=1, size=1000)  # mean=0, std=1

# Uniform distribution
uniform = np.random.uniform(low=0, high=10, size=100)

# Set seed for reproducibility (like in tests)
np.random.seed(42)
```

---

## Integration: Pydantic + Pandas + NumPy

Here's how these libraries work together in real-world scenarios:

### Example 1: Data Pipeline with Validation

```python
from pydantic import BaseModel, validator
import pandas as pd
import numpy as np
from typing import List
from datetime import datetime

class SalesRecord(BaseModel):
    date: datetime
    product_id: int
    quantity: int
    price: float
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v
    
    @validator('price')
    def validate_price(cls, v):
        if v < 0:
            raise ValueError('Price cannot be negative')
        return v

# Validate incoming data
raw_data = [
    {"date": "2024-01-01", "product_id": 1, "quantity": 5, "price": 29.99},
    {"date": "2024-01-02", "product_id": 2, "quantity": 3, "price": 49.99},
    {"date": "2024-01-03", "product_id": 1, "quantity": 2, "price": 29.99},
]

# Validate with Pydantic
validated_records = [SalesRecord(**record) for record in raw_data]

# Convert to Pandas DataFrame
df = pd.DataFrame([record.dict() for record in validated_records])

# Add calculated column
df['total'] = df['quantity'] * df['price']

# NumPy operations for analysis
total_revenue = np.sum(df['total'].values)
avg_quantity = np.mean(df['quantity'].values)
revenue_std = np.std(df['total'].values)

print(f"Total Revenue: ${total_revenue:.2f}")
print(f"Average Quantity: {avg_quantity:.2f}")
print(f"Revenue Std Dev: ${revenue_std:.2f}")
```

### Example 2: Data Analysis with Type Safety

```python
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict

class AnalysisResult(BaseModel):
    metric: str
    value: float
    unit: str

class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def compute_statistics(self, column: str) -> List[AnalysisResult]:
        """Compute statistics with validated output"""
        data = self.df[column].values
        
        results = [
            AnalysisResult(metric="mean", value=float(np.mean(data)), unit=""),
            AnalysisResult(metric="median", value=float(np.median(data)), unit=""),
            AnalysisResult(metric="std", value=float(np.std(data)), unit=""),
            AnalysisResult(metric="min", value=float(np.min(data)), unit=""),
            AnalysisResult(metric="max", value=float(np.max(data)), unit=""),
        ]
        
        return results
    
    def filter_outliers(self, column: str, n_std: float = 2) -> pd.DataFrame:
        """Remove outliers using NumPy calculations"""
        data = self.df[column].values
        mean = np.mean(data)
        std = np.std(data)
        
        mask = np.abs(data - mean) <= n_std * std
        return self.df[mask]

# Usage
df = pd.DataFrame({
    'value': [10, 12, 11, 13, 100, 14, 12, 15],  # 100 is outlier
    'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
})

analyzer = DataAnalyzer(df)
stats = analyzer.compute_statistics('value')

for stat in stats:
    print(f"{stat.metric}: {stat.value:.2f}")

# Remove outliers
clean_df = analyzer.filter_outliers('value')
print(f"\nOriginal rows: {len(df)}, After filtering: {len(clean_df)}")
```

### Example 3: ETL Pipeline

```python
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional

class UserEvent(BaseModel):
    user_id: int
    event_type: str
    timestamp: datetime
    value: Optional[float] = None

class UserMetrics(BaseModel):
    user_id: int
    total_events: int
    unique_event_types: int
    total_value: float
    avg_value: float
    first_event: datetime
    last_event: datetime

def process_events(events: List[dict]) -> pd.DataFrame:
    """ETL pipeline with validation"""
    
    # Extract & Validate
    validated = [UserEvent(**event) for event in events]
    
    # Transform to DataFrame
    df = pd.DataFrame([e.dict() for e in validated])
    
    # Load/Aggregate
    metrics = df.groupby('user_id').agg({
        'event_type': ['count', 'nunique'],
        'value': ['sum', 'mean'],
        'timestamp': ['min', 'max']
    }).reset_index()
    
    # Flatten column names
    metrics.columns = [
        'user_id', 'total_events', 'unique_event_types',
        'total_value', 'avg_value', 'first_event', 'last_event'
    ]
    
    # Fill NaN values
    metrics['total_value'] = metrics['total_value'].fillna(0)
    metrics['avg_value'] = metrics['avg_value'].fillna(0)
    
    # Validate output
    result = [UserMetrics(**row) for row in metrics.to_dict('records')]
    
    return result

# Example usage
events = [
    {"user_id": 1, "event_type": "login", "timestamp": "2024-01-01T10:00:00", "value": None},
    {"user_id": 1, "event_type": "purchase", "timestamp": "2024-01-01T11:00:00", "value": 99.99},
    {"user_id": 2, "event_type": "login", "timestamp": "2024-01-01T09:00:00", "value": None},
    {"user_id": 1, "event_type": "logout", "timestamp": "2024-01-01T12:00:00", "value": None},
]

metrics = process_events(events)
for m in metrics:
    print(f"User {m.user_id}: {m.total_events} events, ${m.total_value:.2f} total value")
```

---

## Key Takeaways for TypeScript Developers

1. **Type Safety**: Use Pydantic for runtime validation (like Zod), type hints for static typing
2. **Data Manipulation**: Pandas is your SQL/lodash for tabular data
3. **Performance**: NumPy for numerical operations - much faster than Python loops
4. **No Null/Undefined**: Use `None` and `Optional[T]` for nullable values
5. **Indentation Matters**: No curly braces, indentation defines code blocks
6. **List Comprehensions**: Python's shorthand for map/filter operations
7. **f-strings**: Template literals in Python (`f"{variable}"`)
8. **Everything is an Object**: But dynamic typing by default (add hints!)

## Common Gotchas

```python
# 1. Mutable default arguments (DON'T DO THIS)
def bad_function(items=[]):  # BAD!
    items.append(1)
    return items

# Do this instead:
def good_function(items=None):
    if items is None:
        items = []
    items.append(1)
    return items

# 2. Integer division
print(5 / 2)   # 2.5 (float division)
print(5 // 2)  # 2 (integer division)

# 3. Copying DataFrames/Arrays
df2 = df  # This is a reference, not a copy!
df2 = df.copy()  # Actual copy

arr2 = arr  # Reference
arr2 = arr.copy()  # Copy

# 4. Chained assignment warning in Pandas
# df[df['age'] > 30]['salary'] = 100000  # BAD - might not work
df.loc[df['age'] > 30, 'salary'] = 100000  # GOOD
```

---

Happy coding! 🐍
basicsbasedpyright: Type annotation for attribute `model_config` is required because this class is not decorated with `@final` [reportUnannotatedClassAttribute]
p