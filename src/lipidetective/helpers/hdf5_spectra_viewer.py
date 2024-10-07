import argparse
import h5py
import pandas as pd
import holoviews as hv
import panel as pn
hv.extension('bokeh')


class SpectraViewer:
    def __init__(self, hdf5_file_path):
        pn.state.on_session_created(self.created)

        self.hdf5_file_path = hdf5_file_path
        self.experiment = h5py.File(hdf5_file_path, "r")
        self.root_keys = list(self.experiment.keys())
        self.select_root_group = pn.widgets.Select(options=self.root_keys)

        self.selection_column = pn.Column(width=360)
        self.selection_column.append(self.select_root_group)

        self.add_group_selection(list(self.experiment.values())[0], 2)

        self.current_dataset_path = '/'.join([column.value for column in self.selection_column])

        self.plot = self.plot_spectrum()
        self.plot_pane = pn.pane.HoloViews(self.plot)
        self.spectrum_info = pn.widgets.Tabulator(self.update_spectrum_info(), name='Spectrum Attributes',
                                                  show_index=True, layout="fit_data_table", disabled=True,
                                                  header_align='center', text_align='center',
                                                  configuration={'columnDefaults': {'headerSort': False}},
                                                  titles={0: 'Metadata', 'index': 'Attributes'}, align='center')

        self.folder_info = pn.pane.Alert(self.update_folder_info(), alert_type='primary', width=300)

        self.app = pn.serve(
            {"SpectraViewer": self.main_app},
            port=61314,
            title="SpectraViewer",
            show=True,
            start=True,
            autoreload=False,
            threaded=True,
        )

        for idx, selector in enumerate(self.selection_column):
            selector.name = f"selector_{idx+1}"
            selector.param.watch(self.change_group_selection, 'value')

    def created(self, session_context):
        print("SpectraViewer session created.")

    def destroyed(self, session_context):
        print("SpectraViewer session destroyed.")
        self.experiment.close()
        self.app.stop()

    def main_app(self):
        pn.state.on_session_destroyed(self.destroyed)
        print(self.hdf5_file_path)

        title = pn.pane.Markdown('''# HDF5 Spectra Viewer''')
        plot_column = pn.Column(self.plot_pane, self.spectrum_info)
        folder_column = pn.Column(self.selection_column, self.folder_info, width=320)
        select = pn.Row(folder_column, plot_column)

        return pn.Column(title, select)

    def add_group_selection(self, group, level_nr):
        if isinstance(group, h5py.Group):
            group_names = list(group.keys())
            first_value = list(group.values())[0]

            # If the next instance is also an HDF5 group, we need to add the current one and also go one level deeper
            if isinstance(first_value, h5py.Group):
                next_selection_level = pn.widgets.Select(options=group_names)
                next_selection_level.name = f"selector_{level_nr}"

                # If we currently have less selectors than the level we are at, we need to add another one
                if len(self.selection_column) < level_nr:
                    self.selection_column.append(next_selection_level)

                else:
                    # If we have enough selectors, we just replace the one we are currently looking at with the new one
                    self.selection_column[level_nr-1] = next_selection_level

                self.add_group_selection(first_value, level_nr + 1)
                next_selection_level.param.watch(self.change_group_selection, 'value')

            # In case the next level is not an HDF5 group anymore also take care of the last level
            else:
                final_selection_level = pn.widgets.Select(options=list(group_names), size=10)
                final_selection_level.name = f"selector_{level_nr}"
                final_selection_level.param.watch(self.change_group_selection, 'value')

                # If the selection column does not provide enough selectors, we need to add another level and selector
                if len(self.selection_column) < level_nr:
                    self.selection_column.append(final_selection_level)

                # In case the final selection level is at least as long as necessary, we exchange the current one on the
                # final level and remove the selectors we don't need anymore.
                else:
                    self.selection_column[level_nr-1] = final_selection_level
                    for selector in self.selection_column[level_nr:]:
                        self.selection_column.remove(selector)

    def change_group_selection(self, event):
        group_index = int(event.obj.name.split('_')[1])
        group_path = '/'.join([column.value for column in self.selection_column[:group_index]])

        self.add_group_selection(self.experiment[group_path], group_index+1)

        self.current_dataset_path = '/'.join([column.value for column in self.selection_column])
        self.plot_pane.object = self.plot_spectrum()
        self.spectrum_info.value = self.update_spectrum_info()
        self.folder_info.object = self.update_folder_info()

    def plot_spectrum(self):
        spectrum = self.experiment[self.current_dataset_path]
        file_title = self.current_dataset_path
        experiment_title = file_title
        mz, intensity = spectrum

        spikes = hv.Spikes((mz, intensity), 'Mass', 'Intensity')
        spikes.opts(xlabel='m/z', width=800, height=400, color='#30a2da')

        return spikes

    def update_spectrum_info(self):
        spectrum = self.experiment[self.current_dataset_path]
        attribute_dict = dict(spectrum.attrs.items())
        attribute_df = pd.DataFrame([list(attribute_dict.values())], columns=list(attribute_dict.keys()))
        return attribute_df.T

    def update_folder_info(self):
        options_len = len(self.selection_column[-1].options)
        return f"\# of spectra in this group: **{options_len}**"

    def close_app(self, event):
        self.experiment.close()
        self.app.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-h5', '--hdf5_file_path', type=str, required=True)
    arguments = parser.parse_args()

    SpectraViewer(arguments.hdf5_file_path)


