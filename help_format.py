from argparse import *

class CustomHelpFormatter(RawTextHelpFormatter):

    def _format_args(self, action, default_metavar):
        get_metavar = self._metavar_formatter(action, default_metavar)
        if action.nargs is None:
            result = '%s' % get_metavar(1)
        elif action.nargs == OPTIONAL:
            result = '[%s]' % get_metavar(1)
        elif action.nargs == ZERO_OR_MORE:
            metavar = get_metavar(1)
            if len(metavar) == 2:
                result = '[%s [%s ...]]' % metavar
            else:
                result = '[%s ...]' % metavar
        elif action.nargs == ONE_OR_MORE:
            result = '%s ...' % get_metavar(1)
        elif action.nargs == REMAINDER:
            result = '...'
        elif action.nargs == PARSER:
            result = '%s ...' % get_metavar(1)
        elif action.nargs == SUPPRESS:
            result = ''
        else:
            try:
                formats = ['%s' for _ in range(action.nargs)]
            except TypeError:
                raise ValueError("invalid nargs value") from None
            result = ' '.join(formats) % get_metavar(action.nargs)
        return result


    def _format_action_invocation(self, action):
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            metavar, = self._metavar_formatter(action, default)(1)
            return metavar

        else:
            parts = []

            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)
                parts = ', '.join(parts)

            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    parts.append('%s' % (option_string))
                parts = '{} {}'.format(', '.join(parts), args_string)

            return parts
