from nicegui import ui, app
import logging
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize counter for demo
count = 0

# Define the main page
@ui.page('/')
def main_page():
    """Main page of the NiceGUI application."""
    with ui.card().classes('w-full max-w-3xl mx-auto'):
        ui.label('NiceGUI Application').classes('text-2xl font-bold')
        ui.markdown('''
        Welcome to the NiceGUI application! NiceGUI allows you to create web UIs with Python.
        
        Features:
        - Build UIs with pure Python
        - Real-time updates
        - Easy integration with FastAPI
        - Interactive components
        ''')
        
        # Counter demo
        with ui.row().classes('items-center'):
            ui.label('Counter:').classes('mr-2')
            label = ui.label(str(count)).classes('text-lg font-bold')
            
            def increment():
                global count
                count += 1
                label.text = str(count)
                logger.info(f"Counter incremented to {count}")
            
            def decrement():
                global count
                count -= 1
                label.text = str(count)
                logger.info(f"Counter decremented to {count}")
            
            ui.button('Decrement', on_click=decrement).props('color=red')
            ui.button('Increment', on_click=increment).props('color=green')
        
        # Add a chart for demonstration
        with ui.card().classes('w-full mt-4'):
            ui.label('Sample Chart').classes('text-xl')
            chart = ui.chart({
                'title': {'text': 'Sample Data'},
                'xAxis': {'categories': ['Jan', 'Feb', 'Mar', 'Apr', 'May']},
                'series': [{
                    'name': 'Data Series 1',
                    'data': [29, 71, 106, 129, 144]
                }, {
                    'name': 'Data Series 2',
                    'data': [80, 120, 105, 110, 95]
                }]
            }).classes('h-64')

# API endpoints can be added with FastAPI
@app.get('/api/health')
def health_check():
    """Health check endpoint."""
    return {'status': 'ok'}

# Configure app
app.title = 'NiceGUI Application'

# This is needed for the main.py integration
if __name__ == '__main__':
    ui.run()