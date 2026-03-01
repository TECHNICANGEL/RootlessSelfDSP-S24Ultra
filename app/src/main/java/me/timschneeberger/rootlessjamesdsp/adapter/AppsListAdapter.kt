package me.timschneeberger.rootlessjamesdsp.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Filter
import android.widget.Filterable
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.RecyclerView
import me.timschneeberger.rootlessjamesdsp.R
import me.timschneeberger.rootlessjamesdsp.model.AppInfo

class AppsListAdapter: RecyclerView.Adapter<AppsListAdapter.ViewHolder>(), Filterable {
    var dataList: List<AppInfo> = emptyList()
        set(value) {
            field = value
            // PERFORMANCE: Use DiffUtil to calculate minimal changes instead of full rebind
            updateFilteredList(value)
        }

    private var filteredDataList: List<AppInfo> = emptyList()

    // PERFORMANCE: Use DiffUtil for efficient RecyclerView updates
    private fun updateFilteredList(newList: List<AppInfo>) {
        val diffCallback = AppInfoDiffCallback(filteredDataList, newList)
        val diffResult = DiffUtil.calculateDiff(diffCallback, true)
        filteredDataList = newList
        diffResult.dispatchUpdatesTo(this)
    }

    inner class ViewHolder(rootView: ViewGroup): RecyclerView.ViewHolder(rootView), View.OnClickListener {

        init { rootView.setOnClickListener(this) }

        private val titleView = rootView.findViewById<TextView>(android.R.id.title)
        private val summaryView = rootView.findViewById<TextView>(android.R.id.summary)
        private val iconView = rootView.findViewById<ImageView>(android.R.id.icon)

        var data: AppInfo? = null
            set(value) {
                field = value
                value ?: return
                titleView.text = value.appName
                summaryView.text = value.packageName
                iconView.setImageDrawable(value.icon)
            }

        override fun onClick(v: View) {
            data?.let {
                onItemClickListener?.onItemClick(it)
            }
        }
    }

    fun interface OnItemClickListener {
        fun onItemClick(appInfo: AppInfo)
    }

    private var onItemClickListener: OnItemClickListener? = null

    fun setOnItemClickListener(listener: OnItemClickListener) {
        this.onItemClickListener = listener
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
        ViewHolder(
            LayoutInflater.from(parent.context)
                .inflate(R.layout.item_app_list, parent, false) as ViewGroup
        )

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        if(position >= itemCount)
            return
        holder.data = filteredDataList[position]
    }

    override fun getItemCount(): Int = filteredDataList.size

    override fun getFilter(): Filter {
        return object : Filter() {
            override fun performFiltering(constraint: CharSequence?): FilterResults {
                val charString = constraint?.toString() ?: ""
                val filtered =
                    if (charString.isEmpty())
                        dataList
                    else {
                        dataList.filter {
                            it.appName.contains(constraint!!, true)
                        }
                    }
                return FilterResults().apply { values = filtered }
            }

            @Suppress("UNCHECKED_CAST")
            override fun publishResults(constraint: CharSequence?, results: FilterResults?) {
                val newList = if (results?.values == null)
                    emptyList()
                else
                    results.values as List<AppInfo>
                // PERFORMANCE: Use DiffUtil instead of notifyDataSetChanged()
                updateFilteredList(newList)
            }
        }
    }

    // PERFORMANCE: DiffUtil callback for efficient list updates
    private class AppInfoDiffCallback(
        private val oldList: List<AppInfo>,
        private val newList: List<AppInfo>
    ) : DiffUtil.Callback() {

        override fun getOldListSize(): Int = oldList.size

        override fun getNewListSize(): Int = newList.size

        override fun areItemsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean {
            // Items are the same if they have the same package name (unique identifier)
            return oldList[oldItemPosition].packageName == newList[newItemPosition].packageName
        }

        override fun areContentsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean {
            // Contents are the same if app name and package name match
            val oldItem = oldList[oldItemPosition]
            val newItem = newList[newItemPosition]
            return oldItem.appName == newItem.appName &&
                   oldItem.packageName == newItem.packageName
        }
    }
}
