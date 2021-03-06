# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from ._model import Model
from ._customwidgets import (EntryPopup, ComboboxPopup, TextPopup)
import psutil
import os
import subprocess
import threading
import queue
import tempfile
import tkinter as tk
from tkinter import messagebox
import tkinter.filedialog as tkfd
import json
from ._uipreferences import UIPreferences
import ast
import pprint
import sys
import logging

logger = logging.getLogger(__name__)


class Controller(object):

    _START, _STOP = 'Start', 'Stop'

    def __init__(self, view):
        self._view = view
        self._model = Model()
        self._filemenu = None
        self._title = tk.StringVar()
        self._sectionsView = None
        self._emptyView = None
        self._sectionView_title = tk.StringVar()
        self._propertiesView = None
        self._textView = None
        self._outputView = None
        self._progress = None
        self._button_text = None
        self._start_button = None
        self._save_algo_json = tk.IntVar()
        self._save_algo_json.set(0)
        self._thread_queue = queue.Queue()
        self._thread = None
        self._command = Controller._START
        self._driver_names = None

        self._process_stop = False
        self._validate_integer_command = self._view.register(Controller._validate_integer)
        self._validate_float_command = self._view.register(Controller._validate_float)

    @property
    def driver_names(self):
        from qiskit_chemistry.drivers import local_drivers
        if self._driver_names is None:
            self._driver_names = local_drivers()

        return self._driver_names

    @staticmethod
    def _validate_integer(action, index, value_if_allowed,
                          prior_value, text, validation_type, trigger_type, widget_name):
        # action=1 -> insert
        if action != '1':
            return True

        if value_if_allowed == '+' or value_if_allowed == '-':
            return True

        try:
            int(value_if_allowed)
            return True
        except ValueError:
            return False

    @staticmethod
    def _validate_float(action, index, value_if_allowed,
                        prior_value, text, validation_type, trigger_type, widget_name):
        # action=1 -> insert
        if action != '1':
            return True

        if value_if_allowed == '+' or value_if_allowed == '-':
            return True

        if value_if_allowed is not None:
            index = value_if_allowed.find('e')
            if index == 0:
                return False

            if index > 0:
                try:
                    float(value_if_allowed[:index])
                except ValueError:
                    return False

                if index < len(value_if_allowed) - 1:
                    right = value_if_allowed[index + 1:]
                    if right == '+' or right == '-':
                        return True
                    try:
                        int(right)
                    except ValueError:
                        return False

                return True

        try:
            float(value_if_allowed)
            return True
        except ValueError:
            return False

    @property
    def outputview(self):
        return self._outputView

    @property
    def model(self):
        if self._model is None:
            self._model = Model()

        return self._model

    def new_input(self):
        try:
            self.stop()
            self._outputView.clear()
            self._start_button.state(['disabled'])
            self._title.set('')
            self._sectionsView.clear()
            self._sectionsView.show_add_button(True)
            self._sectionsView.show_remove_button(False)
            self._textView.clear()
            self._sectionView_title.set('')
            self._propertiesView.clear()
            self._propertiesView.show_remove_button(False)
            self._emptyView.tkraise()

            section_names = self.model.new()
            self._sectionsView.populate(section_names)
            self._start_button.state(['!disabled'])
            missing = self.get_sections_names_missing()
            self._sectionsView.show_add_button(True if missing else False)
            return True
        except Exception as e:
            self._outputView.clear()
            self._outputView.write_line(str(e))

        return False

    def open_file(self, filename):
        try:
            self.stop()
            self._outputView.clear()
            self._start_button.state(['disabled'])
            self._title.set('')
            self._sectionsView.clear()
            self._sectionsView.show_add_button(True)
            self._sectionsView.show_remove_button(False)
            self._textView.clear()
            self._sectionView_title.set('')
            self._propertiesView.clear()
            self._propertiesView.show_remove_button(False)
            self._emptyView.tkraise()

            section_names = self.model.load_file(filename)
            self._title.set(os.path.basename(filename))
            if len(section_names) == 0:
                self._outputView.write_line('No sections found on file')
                return

            self._sectionsView.populate(section_names)
            self._start_button.state(['!disabled'])
            missing = self.get_sections_names_missing()
            self._sectionsView.show_add_button(True if missing else False)
            return True
        except Exception as e:
            self._outputView.clear()
            self._outputView.write_line(str(e))

        return False

    def is_empty(self):
        return self.model.is_empty()

    def save_file(self):
        filename = self.model.get_filename()
        if filename is None or len(filename) == 0:
            self._outputView.write_line("No file to save.")
            return False

        try:
            self.model.save_to_file(filename)
            self._outputView.write_line("Saved file: {}".format(filename))
            return True
        except Exception as e:
            messagebox.showerror("Error", str(e))

        return False

    def save_file_as(self, filename):
        try:
            self.model.save_to_file(filename)
            self.open_file(filename)
            return True
        except Exception as e:
            messagebox.showerror("Error", str(e))

        return False

    def export_dictionary_to_clipboard(self, window):
        try:
            value = json.loads(json.dumps(self.model.get_dictionary()))
            value = pprint.pformat(value, indent=4)
            window.clipboard_clear()
            window.clipboard_append(value)
            self._outputView.write_line("Exported to clibpoard.")
            return dict
        except Exception as e:
            messagebox.showerror("Error", str(e))

        return {}

    def export_dictionary_to_file(self, filename):
        try:
            self.model.export_dictionary(filename)
            self._outputView.write_line(
                "Exported to file: {}".format(filename))
            return True
        except Exception as e:
            messagebox.showerror("Error", str(e))

        return False

    def on_section_select(self, section_name):
        self._sectionsView.show_remove_button(True)
        self._sectionView_title.set(section_name)
        if self.model.section_is_text(section_name):
            text = self.model.get_section_text(section_name)
            self._textView.populate(text)
            self._textView.section_name = section_name
            self._textView.show_add_button(False)
            self._textView.show_remove_button(False)
            self._textView.show_defaults_button(not self.model.default_properties_equals_properties(section_name))
            self._textView.tkraise()
        else:
            self._propertiesView.show_add_button(self.shows_add_button(section_name))
            self._propertiesView.populate(self.model.get_section_properties_with_substitution(section_name))
            self._propertiesView.section_name = section_name
            self._propertiesView.show_remove_button(False)
            self._propertiesView.show_defaults_button(not self.model.default_properties_equals_properties(section_name))
            self._propertiesView.tkraise()

    def on_property_select(self, section_name, property_name):
        from qiskit_aqua.parser import JSONSchema
        _show_remove = property_name != JSONSchema.PROVIDER and property_name != JSONSchema.NAME \
            if section_name == JSONSchema.BACKEND else property_name != JSONSchema.NAME
        self._propertiesView.show_remove_button(_show_remove)

    def on_section_add(self, section_name):
        try:
            if section_name is None:
                section_name = ''
            section_name = section_name.lower().strip()
            if len(section_name) == 0:
                return False

            self.model.set_section(section_name)
            missing = self.get_sections_names_missing()
            self._sectionsView.show_add_button(True if missing else False)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return False

        return True

    def validate_section_add(self, section_name):
        try:
            if section_name in self.model.get_section_names():
                return'Duplicate section name'
        except Exception as e:
            return e.message

        return None

    def on_section_remove(self, section_name):
        try:
            self._sectionsView.show_remove_button(False)
            self.model.delete_section(section_name)
            missing = self.get_sections_names_missing()
            self._sectionsView.show_add_button(True if missing else False)
            self._sectionView_title.set('')
            self._propertiesView.clear()
            self._textView.clear()
            self._emptyView.tkraise()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return False

        return True

    def on_section_defaults(self, section_name):
        from qiskit_chemistry.parser import InputParser
        try:
            self.model.set_default_properties_for_name(section_name)
            if section_name == InputParser.DRIVER:
                section_names = self.model.get_section_names()
                self._sectionsView.populate(section_names)
                missing = self.get_sections_names_missing()
                self._sectionsView.show_add_button(True if missing else False)

            self.on_section_select(section_name)
            return True
        except Exception as e:
            messagebox.showerror("Error", str(e))

        return False

    def get_sections_names_missing(self):
        try:
            section_names = self.model.get_section_names()
            default_sections = self.model.get_default_sections()
            return list(set(default_sections.keys()) - set(section_names))
        except Exception as e:
            self._outputView.write_line(str(e))

    def get_property_names_missing(self, section_name):
        try:
            properties = self.model.get_section_properties(section_name)
            default_properties = self.model.get_section_default_properties(
                section_name)
            if default_properties is None:
                return None
            return list(set(default_properties.keys()) - set(properties.keys()))
        except Exception as e:
            self._outputView.write_line(str(e))

    def shows_add_button(self, section_name):
        if self.model.allows_additional_properties(section_name):
            return True

        missing = self.get_property_names_missing(section_name)
        return missing is None or len(missing) > 0

    def on_property_add(self, section_name, property_name):
        try:
            value = self.model.get_property_default_value(
                section_name, property_name)
            if value is None:
                value = ''

            return self.on_property_set(section_name, property_name, value)
        except Exception as e:
            messagebox.showerror("Error", str(e))

        return False

    def on_property_set(self, section_name, property_name, value):
        from qiskit_aqua.parser import JSONSchema
        try:
            self.model.set_section_property(
                section_name, property_name, value)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return False

        try:
            self._propertiesView.populate(self.model.get_section_properties_with_substitution(section_name))
            self._propertiesView.show_add_button(self.shows_add_button(section_name))
            _show_remove = property_name != JSONSchema.PROVIDER and property_name != JSONSchema.NAME \
                if section_name == JSONSchema.BACKEND else property_name != JSONSchema.NAME
            self._propertiesView.show_remove_button(_show_remove and self._propertiesView.has_selection())
            self._propertiesView.show_defaults_button(not self.model.default_properties_equals_properties(section_name))
            section_names = self.model.get_section_names()
            self._sectionsView.populate(section_names, section_name)
            missing = self.get_sections_names_missing()
            self._sectionsView.show_add_button(True if missing else False)
            return True
        except Exception as e:
            messagebox.showerror("Error", str(e))

        return False

    def validate_property_add(self, section_name, property_name):
        try:
            value = self.model.get_section_property(section_name, property_name)
            if value is not None:
                return 'Duplicate property name'
        except Exception as e:
            return e.message

        return None

    def on_section_property_remove(self, section_name, property_name):
        try:
            self.model.delete_section_property(section_name, property_name)
            self._propertiesView.populate(self.model.get_section_properties_with_substitution(section_name))
            self._propertiesView.show_add_button(self.shows_add_button(section_name))
            self._propertiesView.show_remove_button(False)
            self._propertiesView.show_defaults_button(not self.model.default_properties_equals_properties(section_name))
        except Exception as e:
            self._outputView.write_line(str(e))

    def on_text_set(self, section_name, value):
        try:
            self.model.set_section_text(section_name, value)
            self._textView.show_defaults_button(not self.model.default_properties_equals_properties(section_name))
        except Exception as e:
            self._outputView.write_line(str(e))
            return False

        return True

    def create_popup(self, section_name, property_name, parent, value):
        from qiskit_chemistry.parser import InputParser
        from qiskit_aqua.parser import JSONSchema
        values = None
        types = ['string']
        combobox_state = 'readonly'
        if InputParser.OPERATOR == section_name and JSONSchema.NAME == property_name:
            values = self.model.get_operator_section_names()
        elif InputParser.DRIVER == section_name and JSONSchema.NAME == property_name:
            values = self.driver_names
        elif JSONSchema.NAME == property_name and Model.is_pluggable_section(section_name):
            values = self.model.get_pluggable_section_names(section_name)
        elif JSONSchema.BACKEND == section_name and \
                (JSONSchema.NAME == property_name or JSONSchema.PROVIDER == property_name):
            values = []
            if JSONSchema.PROVIDER == property_name:
                combobox_state = 'normal'
                for provider, _ in self.model.providers.items():
                    values.append(provider)
            else:
                provider_name = self.model.get_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER)
                values = self.model.providers.get(provider_name, [])
        else:
            values = self.model.get_property_default_values(section_name, property_name)
            types = self.model.get_property_types(section_name, property_name)

        if values is not None:
            value = '' if value is None else str(value)
            values = [str(v) for v in values]
            widget = ComboboxPopup(self, section_name,
                                   property_name,
                                   parent,
                                   exportselection=0,
                                   state=combobox_state,
                                   values=values)
            widget._text = value
            if len(values) > 0:
                if value in values:
                    widget.current(values.index(value))
                else:
                    widget.current(0)

            return widget

        value = '' if value is None else value
        if 'number' in types or 'integer' in types:
            vcmd = self._validate_integer_command if 'integer' in types else self._validate_float_command
            vcmd = (vcmd, '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
            widget = EntryPopup(self,
                                section_name,
                                property_name,
                                parent,
                                value,
                                validate='all',
                                validatecommand=vcmd,
                                state=tk.NORMAL)
            widget.selectAll()
            return widget

        if 'object' in types or 'array' in types:
            try:
                if isinstance(value, str):
                    value = value.strip()
                    if len(value) > 0:
                        value = ast.literal_eval(value)

                if isinstance(value, dict) or isinstance(value, list):
                    value = json.dumps(value, sort_keys=True, indent=4)
            except:
                pass

        widget = TextPopup(self,
                           section_name,
                           property_name,
                           parent,
                           value)
        widget.selectAll()
        return widget

    def toggle(self):
        if self.model.is_empty():
            self._outputView.write_line("Missing Input")
            return

        self._start_button.state(['disabled'])
        self._filemenu.entryconfig(0, state='disabled')
        self._filemenu.entryconfig(1, state='disabled')
        self._filemenu.entryconfig(2, state='disabled')
        self._view.after(100, self._process_thread_queue)
        try:
            if self._command is Controller._START:
                self._outputView.clear()
                filename = None
                if self._save_algo_json.get() != 0:
                    preferences = UIPreferences()
                    filename = tkfd.asksaveasfilename(parent=self._view,
                                                      title='Algorithm Input',
                                                      initialdir=preferences.get_savefile_initialdir())
                    if not filename:
                        self._thread_queue.put(None)
                        self._start_button.state(['!disabled'])
                        self._filemenu.entryconfig(0, state='normal')
                        self._filemenu.entryconfig(1, state='normal')
                        self._filemenu.entryconfig(2, state='normal')
                        return

                    preferences.set_savefile_initialdir(
                        os.path.dirname(filename))
                    preferences.save()

                self._thread = AquaChemistryThread(
                    self.model, self._outputView, self._thread_queue, filename)
                self._thread.daemon = True
                self._thread.start()
            else:
                self.stop()
        except Exception as e:
            self._thread = None
            self._thread_queue.put(None)
            self._outputView.write_line("Failure: {}".format(str(e)))
            self._start_button.state(['!disabled'])
            self._filemenu.entryconfig(0, state='normal')
            self._filemenu.entryconfig(1, state='normal')
            self._filemenu.entryconfig(2, state='normal')

    def stop(self):
        if self._thread is not None:
            stopthread = threading.Thread(target=Controller._stop,
                                          args=(self._thread,),
                                          name='Chemistry stop thread')
            stopthread.daemon = True
            stopthread.start()
            self._outputView.clear_buffer()
            self._thread = None
            self._process_stop = True
            self._thread_queue.put(Controller._STOP)

    @staticmethod
    def _stop(thread):
        try:
            if thread is not None:
                thread.stop()
        except:
            pass

    def _process_thread_queue(self):
        try:
            line = self._thread_queue.get_nowait()
            if line is None:
                return
            elif line is Controller._START:
                self._progress.start(500)
                self._command = Controller._STOP
                self._button_text.set(self._command)
                self._start_button.state(['!disabled'])
            elif line is Controller._STOP:
                if not self._outputView.buffer_empty():
                    # repost stop
                    self._thread_queue.put(Controller._STOP)
                else:
                    self._thread = None
                    self._progress.stop()
                    self._command = Controller._START
                    self._button_text.set(self._command)
                    self._start_button.state(['!disabled'])
                    self._filemenu.entryconfig(0, state='normal')
                    self._filemenu.entryconfig(1, state='normal')
                    self._filemenu.entryconfig(2, state='normal')
                    if self._process_stop:
                        self._process_stop = False
                        self._outputView.write_line('Process stopped.')
                    return

            self._view.update_idletasks()
        except:
            pass

        self._view.after(100, self._process_thread_queue)


class AquaChemistryThread(threading.Thread):

    def __init__(self, model, output, queue, filename):
        super(AquaChemistryThread, self).__init__(name='Chemistry run thread')
        self.model = model
        self._output = output
        self._thread_queue = queue
        self._json_algo_file = filename
        self._popen = None

    def stop(self):
        self._output = None
        self._thread_queue = None
        if self._popen is not None:
            p = self._popen
            self._kill(p.pid)
            p.stdout.close()

    def _kill(self, proc_pid):
        try:
            process = psutil.Process(proc_pid)
            for proc in process.children(recursive=True):
                proc.kill()
            process.kill()
        except Exception as e:
            if self._output is not None:
                self._output.write_line(
                    'Process kill has failed: {}'.format(str(e)))

    def run(self):
        input_file = None
        output_file = None
        temp_input = False
        try:
            qiskit_chemistry_directory = os.path.dirname(
                os.path.realpath(__file__))
            qiskit_chemistry_directory = os.path.abspath(
                os.path.join(qiskit_chemistry_directory, '../qiskit_chemistry_cmd'))
            input_file = self.model.get_filename()
            if input_file is None or self.model.is_modified():
                fd, input_file = tempfile.mkstemp(suffix='.in')
                os.close(fd)
                temp_input = True
                self.model.save_to_file(input_file)

            startupinfo = None
            process_name = psutil.Process().exe()
            if process_name is None or len(process_name) == 0:
                process_name = 'python'
            else:
                if sys.platform == 'win32' and process_name.endswith('pythonw.exe'):
                    path = os.path.dirname(process_name)
                    files = [f for f in os.listdir(path) if f != 'pythonw.exe' and f.startswith(
                        'python') and f.endswith('.exe')]
                    # sort reverse to have the python versions first: python3.exe before python2.exe
                    files = sorted(files, key=str.lower, reverse=True)
                    new_process = None
                    for file in files:
                        p = os.path.join(path, file)
                        if os.path.isfile(p):
                            # python.exe takes precedence
                            if file.lower() == 'python.exe':
                                new_process = p
                                break

                            # use first found
                            if new_process is None:
                                new_process = p

                    if new_process is not None:
                        startupinfo = subprocess.STARTUPINFO()
                        startupinfo.dwFlags = subprocess.STARTF_USESHOWWINDOW
                        startupinfo.wShowWindow = subprocess.SW_HIDE
                        process_name = new_process

            input_array = [process_name, qiskit_chemistry_directory, input_file]
            if self._json_algo_file:
                input_array.extend(['-jo', self._json_algo_file])
            else:
                fd, output_file = tempfile.mkstemp(suffix='.out')
                os.close(fd)
                input_array.extend(['-o', output_file])

            if self._output is not None and logger.getEffectiveLevel() == logging.DEBUG:
                self._output.write('Process: {}\n'.format(process_name))

            self._popen = subprocess.Popen(input_array,
                                           stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT,
                                           universal_newlines=True,
                                           startupinfo=startupinfo)
            if self._thread_queue is not None:
                self._thread_queue.put(Controller._START)
            for line in iter(self._popen.stdout.readline, ''):
                if self._output is not None:
                    self._output.write(str(line))
            self._popen.stdout.close()
            self._popen.wait()
        except Exception as e:
            if self._output is not None:
                self._output.write('Process has failed: {}'.format(str(e)))
        finally:
            self._popen = None
            if self._thread_queue is not None:
                self._thread_queue.put(Controller._STOP)
            try:
                if temp_input and input_file is not None:
                    os.remove(input_file)

                input_file = None
            finally:
                if output_file is not None:
                    os.remove(output_file)
                    output_file = None
