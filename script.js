//modelo secuencial
const modelo = tf.sequential()

async function funcionLineal() {
  let num = parseInt(document.getElementById('num').value)

  modelo.add(tf.layers.dense({ units: 1, inputShape: [1] }))

  modelo.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd',
  })

  const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1])
  const ys = tf.tensor2d([-1, 2, 5, 8, 11, 14], [6, 1])

  await modelo.fit(xs, ys, { epochs: 450 })

  document.getElementById('salida').innerText = modelo.predict(
    tf.tensor2d([num], [1, 1])
  )
}
