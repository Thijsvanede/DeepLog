import argparse
import os

class TextHelpFormatter(argparse.RawTextHelpFormatter):
    """Text formatter for ArgumentParser"""

    def __init__(self, prog, indent_increment=2, width=None):
        """Format text without description on newline"""
        # Bugfix for argparse which does not comput column width properly
        if width is None:
            width = os.get_terminal_size(0).columns

        # Call super with infinite max_help_position
        super(TextHelpFormatter, self).__init__(
            prog,
            indent_increment=indent_increment,
            max_help_position=float('inf'),
            width=width
        )

        # Store max default length
        self._default_max_length = 0
        self.   _dest_max_length = 0
        self.   _help_max_length = 0

    def add_argument(self, action):
        """Override: computes maximum length of default"""
        # Call super
        result = super(TextHelpFormatter, self).add_argument(action)
        # Increment _default_max_length
        if action.default and action.nargs != 0:
            self._default_max_length = max(self._default_max_length,
                                           len(str(action.default)))
        if action.dest:
            self._dest_max_length = max(self._dest_max_length,
                len(str(self._format_args(action, action.dest.upper()))))

        if action.help:
            self._help_max_length = max(self._help_max_length,
                                        len(str(action.help)))

        # Return result
        return result


    def _format_action(self, action):
        """Add (default=<default>) to action if any"""

        # Format actions regularly
        result = super(TextHelpFormatter, self)._format_action(action)

        # Add default if any
        if action.default and action.nargs != 0:
            result = result.split('\n')
            space =  self._help_max_length + self._action_max_length + self._current_indent
            if space + self._default_max_length + 12 <= self._width:
                result[-2] = result[-2] + '{} (default = {:>{width}})'  .format(
                    ' '*max(0, space - len(result[-2])),
                    str(action.default),
                    width=self._default_max_length
                )
            else:
                space = self._width - 12 - self._default_max_length
                result[-1] = result[-1] + '{}(default = {:>{width}})\n'.format(
                    ' '*max(1, space),
                    str(action.default),
                    width=self._default_max_length
                )

            result = '\n'.join(result)
        # Return result
        return result



    def _format_action_invocation(self, action):
        """Format actions by showing name only once"""
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar

        else:
            args_string = ""
            parts = []

            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            else:
                default = action.dest.upper()
                args_string = " "+self._format_args(action, default)
                for option_string in action.option_strings:
                    parts.append(option_string)

            # Join parts
            parts = ', '.join(parts)
            result = "{}{padding}{}".format(parts, args_string, padding=
                ' '*(self._action_max_length-len(parts)-self._dest_max_length-3))

            return result
