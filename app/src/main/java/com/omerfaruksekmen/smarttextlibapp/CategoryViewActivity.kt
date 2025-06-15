package com.omerfaruksekmen.smarttextlibapp

import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class CategoryViewActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_category_view)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.categoryView)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        val categoryName = intent.getStringExtra("categoryName")
        title = "$categoryName Documents"

        val categoryTitleTextView = findViewById<TextView>(R.id.categoryTitleTextView)
        categoryTitleTextView.text = "$categoryName Documents"

        val container = findViewById<LinearLayout>(R.id.documentsContainer)

        if (categoryName != null) {
            val sharedPref = getSharedPreferences("SmartLibraryPrefs", MODE_PRIVATE)
            val docs = sharedPref.getStringSet(categoryName, setOf())?.toMutableSet() ?: mutableSetOf()

            if (docs.isNotEmpty()) {
                for (doc in docs) {
                    val parts = doc.split("||")
                    val title = parts.getOrNull(0) ?: "Untitled"
                    val content = parts.getOrNull(1) ?: "No content"

                    val wrapperLayout = LinearLayout(this).apply {
                        orientation = LinearLayout.VERTICAL
                        setPadding(24, 24, 24, 24)
                        setBackgroundResource(R.drawable.document_border)
                        val params = LinearLayout.LayoutParams(
                            LinearLayout.LayoutParams.MATCH_PARENT,
                            LinearLayout.LayoutParams.WRAP_CONTENT
                        )
                        params.setMargins(0, 0, 0, 48)
                        layoutParams = params
                    }

                    val titleView = TextView(this).apply {
                        text = "Title: $title"
                        textSize = 18f
                        setTextColor(Color.BLACK)
                        setTypeface(null, Typeface.BOLD)
                    }

                    val contentView = TextView(this).apply {
                        text = "Content: $content"
                        textSize = 16f
                        setTextColor(Color.DKGRAY)
                        setPadding(0, 12, 0, 0)
                    }

                    val deleteButton = Button(this).apply {
                        text = "Delete"
                        setBackgroundColor(Color.parseColor("#FF5555"))
                        setTextColor(Color.WHITE)
                        setOnClickListener {
                            val alertDialog = androidx.appcompat.app.AlertDialog.Builder(this@CategoryViewActivity)
                                .setTitle("Delete Confirmation")
                                .setMessage("Are you sure you want to delete this document?")
                                .setPositiveButton("Yes") { dialog, _ ->
                                    val updatedDocs = docs.toMutableSet()
                                    updatedDocs.remove(doc)
                                    with(sharedPref.edit()) {
                                        putStringSet(categoryName, updatedDocs)
                                        apply()
                                    }
                                    recreate()
                                    dialog.dismiss()
                                }
                                .setNegativeButton("Cancel") { dialog, _ ->
                                    dialog.dismiss()
                                }
                                .create()

                            alertDialog.show()
                        }

                    }

                    wrapperLayout.addView(titleView)
                    wrapperLayout.addView(contentView)
                    wrapperLayout.addView(deleteButton)
                    container.addView(wrapperLayout)
                }
            } else {
                val noDocsView = TextView(this).apply {
                    text = "No documents found for this category."
                    textSize = 16f
                    setTextColor(Color.GRAY)
                }
                container.addView(noDocsView)
            }
        } else {
            val errorView = TextView(this).apply {
                text = "Invalid category selected."
                textSize = 16f
                setTextColor(Color.RED)
            }
            container.addView(errorView)
        }
    }
}