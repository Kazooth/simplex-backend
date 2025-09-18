from flask import Flask, jsonify, send_file
import io
from tejidos_lp_simplex_grafico import resolver_simplex_max, resolver_simplex_min, vertices_factibles, Z, graficar, imprimir_conclusion
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)

@app.route('/api/optimizacion')
def optimizacion():
    opt_max = resolver_simplex_max()
    opt_min = resolver_simplex_min()
    verts = vertices_factibles()
    rows = [{'x': x, 'y': y, 'Z': Z(x, y)} for (x, y) in verts]
    df = pd.DataFrame(rows).sort_values('Z', ascending=False).reset_index(drop=True)
    return jsonify({
        'max': {'x': opt_max.x, 'y': opt_max.y, 'Z': opt_max.z, 'status': opt_max.status, 'mensaje': opt_max.mensaje},
        'min': {'x': opt_min.x, 'y': opt_min.y, 'Z': opt_min.z, 'status': opt_min.status, 'mensaje': opt_min.mensaje},
        'vertices': df.to_dict(orient='records')
    })

@app.route('/api/grafico')
def grafico():
    opt_max = resolver_simplex_max()
    verts = vertices_factibles()
    plt.clf()
    graficar(verts, opt_max)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
