package com.omerfaruksekmen.smarttextlibapp

import android.content.Intent
import android.content.res.AssetFileDescriptor
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class SmartLibraryActivity : AppCompatActivity() {

    private lateinit var tfidfVocab: List<String>
    private lateinit var idfValues: FloatArray
    private lateinit var vocabIndex: Map<String, Int>
    private lateinit var dlModel: Interpreter
    private lateinit var mlModel: Interpreter
    private var predictionExplanation: String = ""
    private var modelResults: String = ""

    private val NUM_FEATURES = 5000
    private val NUM_CLASSES = 4
    private val CLASS_NAMES = arrayOf("World", "Sports", "Business", "Sci/Tech")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_smart_library)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        val (vocabList, idfArr) = loadVocabAndIdf("tfidf_full.json")
        tfidfVocab = vocabList
        idfValues = idfArr
        vocabIndex = tfidfVocab.mapIndexed { index, token -> token to index }.toMap()

        dlModel = Interpreter(loadModelFile("dl_model.tflite"))
        mlModel = Interpreter(loadModelFile("ml_model.tflite"))

        val titleInput = findViewById<EditText>(R.id.titleEditText)
        val bodyInput = findViewById<EditText>(R.id.contentEditText)
        val predictButton = findViewById<Button>(R.id.predictButton)
        val saveButton = findViewById<Button>(R.id.saveButton)
        val resultText = findViewById<TextView>(R.id.predictionTextView)
        val analysisButton = findViewById<Button>(R.id.analysisButton)
        analysisButton.visibility = View.GONE

        val categoryButtons = mapOf(
            R.id.btnWorld to "World",
            R.id.btnSports to "Sports",
            R.id.btnBusiness to "Business",
            R.id.btnSciTech to "Sci/Tech"
        )

        categoryButtons.forEach { (buttonId, category) ->
            findViewById<Button>(buttonId).setOnLongClickListener {
                val intent = Intent(this, CategoryViewActivity::class.java)
                intent.putExtra("categoryName", category)
                startActivity(intent)
                true
            }
        }

        var predictedCategory: String? = null
        var predictionDetails: String = ""

        predictButton.setOnClickListener {
            val title = titleInput.text.toString()
            val body = bodyInput.text.toString()
            if (title.isBlank() || body.isBlank()) {
                resultText.text = "Please enter both title and body."
                return@setOnClickListener
            }

            val fullText = "$title $body"

            // 1. Metin özellikleri
            val wordCount = fullText.split("\\s+".toRegex()).size
            val charCount = fullText.length

            // 2. Ön işleme
            val cleanedText = fullText.lowercase()
                .replace("[^a-zA-Z0-9\\s]".toRegex(), "")
                .replace("\\s+".toRegex(), " ").trim()

            // 3. TF-IDF vektörizasyon
            val inputVec = textToTfidfVector(fullText)

            // En yüksek 10 TF-IDF değerini ve karşılık gelen kelimeleri bul
            val topIndicesWithScores = inputVec
                .mapIndexed { index, value -> index to value }
                .filter { it.second > 0f }
                .sortedByDescending { it.second }
                .take(10)

            val topTfidfWords = topIndicesWithScores.map { (index, score) ->
                val word = tfidfVocab.getOrNull(index) ?: "N/A"
                "$word: %.4f".format(score)
            }.joinToString("\n")

            val (dlClass, dlConfidence) = predictWithModel(dlModel, inputVec)
            val (mlClass, mlConfidence) = predictWithModel(mlModel, inputVec)

            val (finalClass, finalConfidence, source) =
                if (dlConfidence >= mlConfidence) {
                    Triple(dlClass, dlConfidence, "Deep Learning")
                } else {
                    Triple(mlClass, mlConfidence, "Logistic Regression")
                }

            predictedCategory = CLASS_NAMES[finalClass]
            predictionDetails = """
        DL Model (FNN): ${CLASS_NAMES[dlClass]} (${(dlConfidence * 100).toInt()}%)
        ML Model (Logistic Regression): ${CLASS_NAMES[mlClass]} (${(mlConfidence * 100).toInt()}%)
        Final Decision: $predictedCategory (from $source model)
    """.trimIndent()

            resultText.text = predictionDetails

            // 5. Tahmin süreci açıklama metni
            val tfidfDetails = inputVec.joinToString(
                prefix = "[",
                postfix = "]",
                limit = 20
            ) { "%.3f".format(it) }

            modelResults = "Deep Learning Prediction: ${CLASS_NAMES[dlClass]} (${(dlConfidence * 100).toInt()}%)\nLogistic Regression Prediction: ${CLASS_NAMES[mlClass]} (${(mlConfidence * 100).toInt()}%)\nFinal Chosen Category: ${CLASS_NAMES[finalClass]} (via $source)"
            predictionExplanation = ("Prediction Analysis:\n\nRaw Title: $title\n\nRaw Body: $body\n\nRaw Body Length: $charCount characters\n\nWord Count: $wordCount\n\nCleaned Text: $cleanedText\n\nTop TF-IDF Words:\n$topTfidfWords\n\n" + modelResults).trimIndent()

            analysisButton.visibility = View.VISIBLE
        }

        analysisButton.setOnClickListener {
            val intent = Intent(this, AnalysisActivity::class.java)
            intent.putExtra("explanation", predictionExplanation)
            startActivity(intent)
        }

        saveButton.setOnClickListener {
            val title = titleInput.text.toString()
            val body = bodyInput.text.toString()
            if (title.isBlank() || body.isBlank()) {
                resultText.text = "Please enter both title and body."
                return@setOnClickListener
            }

            if (predictedCategory == null) {
                resultText.text = "Please perform prediction first."
                return@setOnClickListener
            }

            val sharedPref = getSharedPreferences("SmartLibraryPrefs", MODE_PRIVATE)
            val key = predictedCategory!!
            val existing = sharedPref.getStringSet(key, mutableSetOf())?.toMutableSet() ?: mutableSetOf()
            existing.add("$title||$body")

            with(sharedPref.edit()) {
                putStringSet(key, existing)
                apply()
            }

            resultText.text = "Document saved to $predictedCategory category.\n$predictionDetails"
        }
    }

    private fun loadVocabAndIdf(filename: String): Pair<List<String>, FloatArray> {
        val jsonStr = assets.open(filename).bufferedReader().use { it.readText() }
        val jsonObject = JSONObject(jsonStr)
        val vocabJsonArray = jsonObject.getJSONArray("vocab")
        val idfJsonArray = jsonObject.getJSONArray("idf")

        val vocabList = mutableListOf<String>()
        for (i in 0 until vocabJsonArray.length()) {
            vocabList.add(vocabJsonArray.getString(i))
        }

        val idfArray = FloatArray(idfJsonArray.length())
        for (i in 0 until idfJsonArray.length()) {
            idfArray[i] = idfJsonArray.getDouble(i).toFloat()
        }

        return Pair(vocabList, idfArray)
    }

    private fun loadModelFile(filename: String): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = assets.openFd(filename)
        val inputStream = fileDescriptor.createInputStream()
        val fileChannel: FileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun textToTfidfVector(text: String): FloatArray {
        val tokens = text.lowercase().split("\\W+".toRegex()).filter { it.isNotBlank() }
        val vec = FloatArray(NUM_FEATURES) { 0f }
        val tokenCounts = mutableMapOf<String, Int>()

        tokens.forEach { token ->
            tokenCounts[token] = tokenCounts.getOrDefault(token, 0) + 1
        }

        val totalTokens = tokens.size.toFloat()

        for ((token, count) in tokenCounts) {
            val index = vocabIndex[token]
            if (index != null && index in 0 until NUM_FEATURES) {
                val tf = count.toFloat() / totalTokens
                val idf = idfValues[index]
                vec[index] = tf * idf
            }
        }
        return vec
    }

    private fun predictWithModel(model: Interpreter, inputVec: FloatArray): Pair<Int, Float> {
        val input = arrayOf(inputVec)
        val output = Array(1) { FloatArray(NUM_CLASSES) }
        model.run(input, output)

        val confidences = output[0]
        val predictedIndex = confidences.indices.maxByOrNull { confidences[it] } ?: -1
        return Pair(predictedIndex, confidences[predictedIndex])
    }
}