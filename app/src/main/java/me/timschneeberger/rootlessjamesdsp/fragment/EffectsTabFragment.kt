package me.timschneeberger.rootlessjamesdsp.fragment

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import me.timschneeberger.rootlessjamesdsp.R
import timber.log.Timber

/**
 * Effects Tab - Container for existing DSP effects fragment
 */
class EffectsTabFragment : Fragment() {

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.tab_effects, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Only add DspFragment on first creation - not after configuration change
        // When restored, childFragmentManager already has the DspFragment
        if (savedInstanceState == null) {
            childFragmentManager.beginTransaction()
                .replace(R.id.effects_container, DspFragment.newInstance())
                .commit()
        }

        Timber.d("EffectsTabFragment initialized")
    }

    companion object {
        fun newInstance() = EffectsTabFragment()
    }
}
