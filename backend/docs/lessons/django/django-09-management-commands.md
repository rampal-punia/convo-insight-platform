# Quick Win 09: Management Commands — Custom CLI Tools

> Write your own `python manage.py` commands for ops, data seeding, and maintenance.

---

## What Are Management Commands?

CLI scripts that run within the Django environment. They have access to models, settings, and everything else.

```
python manage.py seed_demo
python manage.py create_random_users 10
python manage.py train_intent_model
```

---

## Creating a Management Command

### File structure:

```
apps/myapp/
└── management/
    ├── __init__.py
    └── commands/
        ├── __init__.py
        └── my_command.py
```

### Basic template:

```python
# apps/myapp/management/commands/my_command.py
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Description of what this command does'

    def handle(self, *args, **options):
        self.stdout.write('Hello from my command!')
        self.stdout.write(self.style.SUCCESS('Done!'))
```

Run it: `python manage.py my_command`

---

## Adding Arguments

### Named arguments (flags):

```python
# apps/dashboard/management/commands/seed_demo.py
class Command(BaseCommand):
    help = 'Seed the database with demo data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--users', type=int, default=10,
            help='Number of users to create'
        )
        parser.add_argument(
            '--orders', type=int, default=20,
            help='Number of orders to create'
        )
        parser.add_argument(
            '--reset', action='store_true',
            help='Delete existing data before seeding'
        )

    @transaction.atomic
    def handle(self, *args, **opts):
        if opts['reset']:
            self._reset()

        users = self._seed_users(opts['users'])
        self._seed_orders(users, opts['orders'])

        self.stdout.write(self.style.SUCCESS('\nDemo data seeded.'))
```

Run with args:
```bash
python manage.py seed_demo --users=50 --orders=100 --reset
```

### Positional arguments:

```python
# apps/llms/management/commands/train_deploy_model.py
def add_arguments(self, parser):
    parser.add_argument('model_type', type=str, help='Model type')
    parser.add_argument('script_path', type=str, help='Training script')
    parser.add_argument('train_data_path', type=str, help='Training data')
    parser.add_argument('output_path', type=str, help='Output directory')
    parser.add_argument('endpoint_name', type=str, help='Endpoint name')
```

Run:
```bash
python manage.py train_deploy_model bert train.py data/ output/ my-endpoint
```

---

## Output Styling

```python
self.stdout.write('Normal message')
self.stdout.write(self.style.SUCCESS('Green success message'))
self.stdout.write(self.style.ERROR('Red error message'))
self.stdout.write(self.style.WARNING('Yellow warning'))
self.stdout.write(self.style.NOTICE('Blue notice'))

# Progress
self.stdout.write(f'Created user {i}', ending='\r')  # Overwrite same line
self.stdout.write('Done!')                             # New line
```

---

## Real Commands in This Project

### `create_random_users` (accounts):

```python
class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('total', type=int, help='Number of users')

    def handle(self, *args, **kwargs):
        total = kwargs['total']
        for i in range(total):
            username = f'user_{i}'
            User.objects.create_user(
                username=username,
                email=f'{username}@example.com',
                password='Aa#123456'
            )
            self.stdout.write(f'User created: {username}')
        self.stdout.write(self.style.SUCCESS(f'Created {total} users'))
```

### `seed_demo` (dashboard):

The most thorough command — uses `@transaction.atomic`, helper methods, and multiple options.

```bash
python manage.py seed_demo --users=5 --orders=10 --reset
```

### `populate_rag_store` (playground):

Populates the vector store with documents for RAG queries.

```bash
python manage.py populate_rag_store
```

---

## Patterns to Know

### `@transaction.atomic`

Wraps the entire command in a database transaction. If anything fails, all changes are rolled back:

```python
@transaction.atomic
def handle(self, *args, **opts):
    # If any of these fail, NONE of them are saved
    self._seed_categories()
    self._seed_products()
    self._seed_orders()
```

### Queryset iteration for large datasets:

```python
def handle(self, *args, **opts):
    # BAD — loads all records into memory
    for order in Order.objects.all():
        process(order)

    # GOOD — processes in chunks of 1000
    for order in Order.objects.iterator(chunk_size=1000):
        process(order)
```

### Calling other commands:

```python
from django.core.management import call_command

def handle(self, *args, **opts):
    call_command('migrate')
    call_command('seed_demo', users=5)
```

---

## Quick Exercise

1. Run `python manage.py seed_demo --help` — see the arguments
2. Run `python manage.py seed_demo --users=3 --orders=5`
3. Create your own management command:
   ```python
   # apps/products/management/commands/count_products.py
   class Command(BaseCommand):
       def handle(self, *args, **options):
           count = Product.objects.filter(is_active=True).count()
           self.stdout.write(f'Active products: {count}')
   ```
4. Add a `--category` filter argument to your command
