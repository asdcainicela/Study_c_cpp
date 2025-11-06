/*
“Cronómetro y Scheduler de Tareas”

Objetivo: crear un programa que ejecute varias tareas simuladas a intervalos fijos y registre su duración con diferentes relojes.

Puntos a implementar:

Duraciones (duration): define distintos intervalos de tareas (500ms, 1s, 2s).

Time points (steady_clock y high_resolution_clock): mide cuánto tarda cada tarea y calcula totales.

Duration cast: convierte entre ms, µs y s para mostrar resultados claros.

Literals: usa 500ms, 1s para definir los intervalos directamente en el código.

Sleep y sleep_until: haz un bucle que ejecute tareas periódicas sin drift, usando sleep_until.

Stopwatch: registra y muestra el tiempo que tarda cada tarea individual y total del bucle.

Timer loop: crea al menos 3 tareas distintas con intervalos diferentes, sincronizadas con steady_clock.

Datetime: imprime la hora del sistema al inicio y al final del programa.

Salida esperada: en consola verás el tick de cada tarea, duración de cada una, tiempo total y comparación entre relojes.

Esto te obliga a usar todo lo aprendido de los archivos 1 a 10, en un proyecto compacto pero completo.
*/